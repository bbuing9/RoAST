import os
import time
import numpy as np
import json
from abc import *

import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import pickle as pkl

from common import DATA_PATH
from datasets import load_dataset
import utils_glue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_dataset(data_name, tokenizer, tokenizer2, seed=0, eval=False, sent=False, adv=False, ood=False):
    # Text Classifications
    if data_name == 'stsb':
        n_class = 1
    elif data_name == 'mnli':
        n_class = 3
    elif data_name == 'news20':
        n_class = 20
    else:
        n_class = 2
    # GLUE TASKs
    if data_name == 'fairmnli':
        dataset = FAIRDataset(data_name, n_class, tokenizer, tokenizer2, eval, seed)
    elif eval:
        dataset = NLIDataset(data_name, n_class, tokenizer, tokenizer2, eval, seed)
    elif sent:
        dataset = SentDataset(data_name, n_class, tokenizer, tokenizer2, sent, seed)
    elif adv:
        dataset = AdvDataset(data_name, n_class, tokenizer, tokenizer2, adv, seed)
    elif ood:
        dataset = OODDataset(data_name, n_class, tokenizer, tokenizer2, ood, seed)
    else:
        dataset = GLUEDataset(data_name, n_class, tokenizer, tokenizer2, eval, seed)

    return dataset

def create_tensor_dataset(inputs, labels, subs, sub_masks, attentions, index):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels)  # (N)
    subs = torch.stack(subs)
    sub_masks = torch.stack(sub_masks)
    attentions = torch.stack(attentions) 
    index = np.array(index)
    index = torch.Tensor(index).long()

    print("Number of samples: {}".format(len(inputs)))

    dataset = TensorDataset(inputs, labels, subs, sub_masks, attentions, index)

    return dataset

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, total_class, tokenizer, substitution_tokenizer, eval, seed=0):

        self.data_name = data_name
        self.total_class = total_class
        self.root_dir = os.path.join(DATA_PATH, data_name)
        self.tokenizer = tokenizer
        self.substitution_tokenizer = substitution_tokenizer
        self.eval = eval

        self.n_classes = int(self.total_class)  # Split a given data
        self.class_idx = list(range(self.n_classes))  # all classes

        if not self._check_exists(self.eval):
            self._preprocess()

        if self.eval:
            self.eval_dataset = torch.load(self._path)
        else:
            self.train_dataset = torch.load(self._train_path)
            self.val_dataset = torch.load(self._val_path)
            self.test_dataset = torch.load(self._test_path)

    @property
    def base_path(self):
        base_path = '{}_{}'.format(self.data_name, self.tokenizer.name)

        return base_path

    @property
    def _path(self):
        return os.path.join(DATA_PATH, self.base_path + '_rift.pth')

    @property
    def _train_path(self):
        return os.path.join(self.root_dir, self.base_path + '_train_rift.pth')

    @property
    def _val_path(self):
        return os.path.join(self.root_dir, self.base_path + '_val_rift.pth')

    @property
    def _test_path(self):
        return os.path.join(self.root_dir, self.base_path + '_test_rift.pth')

    def _check_exists(self, eval):
        if eval:
            if not os.path.exists(self._path) :
                return False
            else:
                return True
        else:
            if not os.path.exists(self._train_path):
                return False
            elif not os.path.exists(self._val_path):
                return False
            elif not os.path.exists(self._test_path):
                return False
            else:
                return True

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass

def get_substitution_dict(file_path):
    import json
    with open(file_path) as f:
        subs_dict = json.load(f)
    return subs_dict

class GLUEDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, substitution_tokenizer, data_ratio=1.0, seed=0):
        super(GLUEDataset, self).__init__(data_name, n_class, tokenizer, substitution_tokenizer, data_ratio, seed)

        self.data_name = data_name 

    def _preprocess(self):
        print('Pre-processing news dataset...')
        train_dataset = self._load_dataset('train')

        if self.data_name == 'mnli':
            val_dataset = self._load_dataset('validation_matched')
            test_dataset = self._load_dataset('validation_mismatched')
        else:
            val_dataset = self._load_dataset('validation')
            test_dataset = val_dataset

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'validation_matched', 'validation_mismatched']

        data_set = load_dataset('glue', self.data_name, split=mode)

        # Get the lists of sentences and their labels.
        inputs, labels, text_subss, text_subs_masks, attention_masks, indices = [], [], [], [], [], []

        # RIFT 
        subs_dict = get_substitution_dict('./attack/counterfitted_neighbors.json')
        tokenized_subs_dict = {} # key is text word, contents are tokenized substitution words

        # Tokenize syn data
        print("tokenizing substitution words")
        for key in subs_dict:
            if len(subs_dict[key])!=0:
                #temp = tokenizer.encode_plus(subs_dict[key], None, add_special_tokens=False, pad_to_max_length=False)['input_ids']
                temp = self.substitution_tokenizer(subs_dict[key])['input_ids']
                #temp = self.tokenizer.encode(subs_dict[key], max_length=128, add_special_tokens=False, padding=False)
                temp = [x[0] for x in temp]
                tokenized_subs_dict[key] = temp

        for i in range(len(data_set)):
            data_n = data_set[i]

            if self.data_name == 'cola' or self.data_name == 'sst2':
                #print(data_n['sentence'])
                encoded = self.tokenizer(data_n['sentence'], 128)
                sent = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                #sent = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=128,
                #                             pad_to_max_length=True, return_tensors='pt')[0]
                #attention_mask = (sent != 1).long()

                text_subs=[]
                text_subs_mask=[]
                for token in sent:
                    text_subs_mask.append([])
                    text_subs.append([token])

                splited_words = data_n['sentence'].split()
                for i in range(min(128 - 2, len(splited_words))):
                    word = splited_words[i]
                    if word in tokenized_subs_dict:
                        text_subs[i+1].extend(tokenized_subs_dict[word])
                
                for i in range(len(sent)):
                    text_subs_mask[i] = [1 for times in range(len(text_subs[i]))]
                    
                    while(len(text_subs[i])<10): # From the official implementation
                        text_subs[i].append(0)
                        text_subs_mask[i].append(0)
            else:
                encoded = self.tokenizer(data_n['premise']+"</s>"+data_n['hypothesis'], 128)

                sent = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                text_subs=[]
                text_subs_mask=[]
                for token in sent:
                    text_subs_mask.append([])
                    text_subs.append([token])

                splited_words = data_n['hypothesis'].split()

                h_start = len(data_n['premise'].split())+2
                len_h = len(splited_words)

                for i in range(h_start, min(128-1, h_start+len_h), 1):
                    word = splited_words[i-h_start]
                    if word in tokenized_subs_dict:
                        text_subs[i].extend(tokenized_subs_dict[word])
                
                for i in range(len(sent)):
                    text_subs_mask[i] = [1 for times in range(len(text_subs[i]))]
                    
                    while(len(text_subs[i])<10): # From the official implementation
                        text_subs[i].append(0)
                        text_subs_mask[i].append(0)
            
            inputs.append(torch.LongTensor(sent).unsqueeze(0))
            #inputs.append(sent.unsqueeze(0))
            labels.append(torch.tensor(data_n['label']).long())
            text_subss.append(torch.tensor(text_subs,dtype=torch.long).unsqueeze(0))
            text_subs_masks.append(torch.tensor(text_subs_mask,dtype=torch.float).unsqueeze(0))
            attention_masks.append(torch.tensor(attention_mask, dtype = torch.long).unsqueeze(0))
            indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, text_subss, text_subs_masks, attention_masks, indices)
        return dataset

class NLIDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, eval, seed=0):
        super(NLIDataset, self).__init__(data_name, n_class, tokenizer, eval, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing ood dataset...')

        data_lists = {"nq-nli": ['dev'], "anli1": ['R1_test'], "anli2": ['R2_test'], "anli3":['R3_test'], "diagnostics": ['diagnostics'], 
                    "epistemic_reasoning": ['test'],  "fever-nli": ['dev'], "hans": ['dev'], "mnli_m": ['dev_matched'], "mnli_mm": ['dev_mismatched'], 
                    "qnli": ['dev'], "wnli": ['train'], "wanli": ['test']}
        data_list = data_lists[self.data_name]

        for data_type in data_list:
            dataset = self._load_dataset(data_type)
            loc = self._path
            torch.save(dataset, loc)

    def _load_dataset(self, mode):
        binary = {'entailment': 0, 'non-entailment': 2}
        ternary = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        loc = './data_preprocess/' + self.data_name + '/' + mode + '.jsonl'
        data_set = pd.read_json(loc, lines=True)
        golds = data_set['gold']
        sents1 = data_set['premise']
        sents2 = data_set['hypothesis']
        
        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(len(data_set)):
            sent1, sent2 = sents1[i], sents2[i]
            if self.data_name == 'anli1':
                length = 512
            else:
                length = 128
            toks = self.tokenizer.encode(sent1, sent2, add_special_tokens=True, max_length=length,
                                             pad_to_max_length=True, return_tensors='pt')

            if 'mnli' in self.data_name or 'anli' in self.data_name or self.data_name == 'diagnostics' or self.data_name == 'fever-nli' or self.data_name == 'wanli':
                if golds[i] != 'entailment' and golds[i] != 'neutral' and  golds[i] != 'contradiction':
                    continue
                label = torch.tensor(ternary[golds[i]]).long()
            else:
                if golds[i] != 'entailment' and golds[i] != 'non-entailment':
                    continue
                label = torch.tensor(binary[golds[i]]).long()

            inputs.append(toks[0])
            labels.append(label)
            indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset

class SentDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, sent, seed=0):
        super(SentDataset, self).__init__(data_name, n_class, tokenizer, sent, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing ood dataset...')

        dataset = self._load_dataset()
        loc = self._path
        torch.save(dataset, loc)

    def _load_dataset(self):
        if self.data_name == 'yelp':
            data_set = load_dataset('yelp_review_full')['test']
            sents = data_set['text']
            labelss = data_set['label']
        elif self.data_name == 'sst2':
            data_set = load_dataset('glue', self.data_name, split='validation')
        elif self.data_name == 'imdb':
            data_set = load_dataset('SetFit/imdb')['test']
            sents = data_set['text']
            labelss = data_set['label']
        else: # cimdb, dynasent1, dynasent2, poem, amazon
            with open("./data_preprocess/{}/merge_text".format(self.data_name), "rb") as output_file:
                sents = pkl.load(output_file)
            with open("./data_preprocess/{}/merge_label".format(self.data_name), "rb") as output_file:
                labelss = pkl.load(output_file)
            data_set = sents
        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(len(data_set)):
            if self.data_name == 'sst2':
                data_n = data_set[i]
                sent = data_n['sentence']
                labels_i = data_n['label']
            else:
                sent = sents[i]
                labels_i = labelss[i]
            toks = self.tokenizer.encode(sent, add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')
            label = torch.tensor(labels_i).long()

            if self.data_name == 'yelp':
                if labelss[i] == 0 or labelss[i] == 4:
                    if labelss[i] == 4:
                        label = torch.tensor(1).long()
                    inputs.append(toks[0])
                    labels.append(label)
                    indices.append(i)
            else:
                inputs.append(toks[0])
                labels.append(label)
                indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset

class AdvDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, adv, seed=0):
        super(AdvDataset, self).__init__(data_name, n_class, tokenizer, adv, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing adv dataset...')

        data_lists = {"advglue_mnli_m": ['mnli'], "advglue_mnli_mm": ['mnli-mm'], "advglue_sst2": ['sst2'],
                      "infobert_mnli_matched": ['infobert_adv_dataset'], "infobert_mnli_mismatched": ['infobert_adv_dataset'], 
                      "roberta_mnli_matched": ['roberta_adv_dataset'], "roberta_mnli_mismatched":['roberta_adv_dataset']}
        data_list = data_lists[self.data_name]

        for data_type in data_list:
            dataset = self._load_dataset(data_type)
            loc = self._path
            torch.save(dataset, loc)

    def _load_dataset(self, mode):
        if 'advglue' in self.data_name:
            f = open('./data_preprocess/adv_glue.json')
            data = json.load(f)
            data_set = data[mode]
        else:
            ternary = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

            loc = './data_preprocess/adv_mnli/' + self.data_name + '/' + mode + '.jsonl'
            data_set = pd.read_json(loc, lines=True)
                
            golds = data_set['true_label']
            sents1 = data_set['orig_preimise']
            sents2 = data_set['adv_hypothesis']
            
        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(len(data_set)):
            if self.data_name == 'advglue_sst2':
                toks = self.tokenizer.encode(data_set[i]['sentence'], add_special_tokens=True, max_length=256,
                                             pad_to_max_length=True, return_tensors='pt')
                label = torch.tensor(data_set[i]['label']).long()
            elif 'advglue' in self.data_name:
                toks = self.tokenizer.encode(data_set[i]['premise'], data_set[i]['hypothesis'], add_special_tokens=True, max_length=256,
                                             pad_to_max_length=True, return_tensors='pt')
                label = torch.tensor(data_set[i]['label']).long()
            else:
                sent1, sent2 = sents1[i], sents2[i]
                toks = self.tokenizer.encode(sent1, sent2, add_special_tokens=True, max_length=128,
                                                pad_to_max_length=True, return_tensors='pt')

                if golds[i] != 'entailment' and golds[i] != 'neutral' and  golds[i] != 'contradiction':
                    continue
                label = torch.tensor(ternary[golds[i]]).long()

            inputs.append(toks[0])
            labels.append(label)
            indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset


class OODDataset(BaseDataset):
    def __init__(self, data_name, n_class, tokenizer, ood, seed=0):
        super(OODDataset, self).__init__(data_name, n_class, tokenizer, ood, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing abnormal dataset...')

        dataset = self._load_dataset()
        loc = self._path
        torch.save(dataset, loc)

    def _load_dataset(self):
        if self.data_name == 'wmt16':
            data_set = load_dataset('wmt16', 'de-en', 'test')
            data_set = data_set['test']['translation']
        elif self.data_name == 'multi30k':
            loc = './data_preprocess/multi30k/multi30k.txt'
            with open(loc) as f:
                data_set = f.readlines()
        elif self.data_name == '20news':
            loc = os.path.join('./data_preprocess/20news/test.csv')    
            with open(loc, encoding='utf-8') as f:
                data_set = f.readlines()
        elif 'mnli' in self.data_name:
            if self.data_name == 'mnli_m':
                mode = 'dev_matched'
            else:
                mode = 'dev_mismatched'
            loc = './data_preprocess/' + self.data_name + '/' + mode + '.jsonl'
            data_set = pd.read_json(loc, lines=True)
            sents1 = data_set['premise']
            sents2 = data_set['hypothesis']
        else:
            data_set = load_dataset('glue', self.data_name, split='validation')
        
        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(len(data_set)):
            if self.data_name == 'wmt16':
                sent = data_set[i]['en']
            elif self.data_name == 'multi30k':
                sent = data_set[i]
            elif self.data_name == 'qqp':
                sent1, sent2 = data_set[i]['question1'], data_set[i]['question2']
                sent = sent1 + sent2
            elif 'mnli' in self.data_name:
                sent1, sent2 = sents1[i], sents2[i]
                sent = sent1 + sent2
            elif self.data_name == '20news':
                # From https://github.com/alinlab/MASKER
                toks = data_set[i].split(',')
                if not int(toks[1]) in self.class_idx:  # only selected classes
                    continue

                path = os.path.join('./data_preprocess/{}'.format(toks[0]))
                with open(path, encoding='utf-8', errors='ignore') as f:
                    sent = f.read()
            else:
                sent = data_set[i]['sentence']    

            toks = self.tokenizer.encode(sent, add_special_tokens=True, max_length=128,
                                             pad_to_max_length=True, return_tensors='pt')

            inputs.append(toks[0])
            labels.append(torch.LongTensor([0]))
            indices.append(i)

        dataset = create_tensor_dataset(inputs, labels, indices)
        return dataset