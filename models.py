import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from transformers import DebertaConfig, DebertaTokenizer, DebertaForSequenceClassification, XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification
from transformers import ElectraConfig, ElectraTokenizer, ElectraForSequenceClassification

def load_models(args):
    if args.backbone == 'bert-large-uncased' or args.backbone == 'bert-base-uncased' :
        config = BertConfig.from_pretrained(args.backbone, num_labels=args.n_class)
    elif 'albert' in args.backbone:
        config = AlbertConfig.from_pretrained(args.backbone, num_labels=args.n_class)
    elif 'deberta' in args.backbone:
        config = DebertaConfig.from_pretrained('microsoft/' + args.backbone, num_labels=args.n_class)
    elif 'xlnet' in args.backbone:
        config = XLNetConfig.from_pretrained(args.backbone, num_labels=args.n_class)
    elif 'gpt' in args.backbone:
        config = GPT2Config.from_pretrained(args.backbone, num_labels=args.n_class)    
    elif 'electra' in args.backbone:
        config = ElectraConfig.from_pretrained('google/' + args.backbone, num_labels=args.n_class)   
    else:
        config = RobertaConfig.from_pretrained(args.backbone, num_labels=args.n_class)

    # Update config
    config.coeff_sym = args.coeff_sym
    config.adv_eps = args.adv_eps

    ## Un-used configs
    config.dropword_prob = 0
    config.droplayer_prob = 0
    config.drophead_prob = 0
    config.hidden_cutoff = 0
    
    config.r3f_noise_eps = 0
    config.r3f_noise_type = 0
    config.saliency_k = 0

    config.ours = None
    config.n_infer = 1
    config.select = 'max'

    config.adv_grad_mask_k = None
    config.adv_grad_mask_large = None
    
    if args.backbone == 'bert-large-uncased' or args.backbone == 'bert-base-uncased' :
        tokenizer = BertTokenizer.from_pretrained(args.backbone, use_fast=True)
        model = BertForSequenceClassification.from_pretrained(args.backbone, config=config)
    elif 'albert' in args.backbone:
        tokenizer = AlbertTokenizer.from_pretrained(args.backbone, use_fast=True)
        model = AlbertForSequenceClassification.from_pretrained(args.backbone, config=config)
    elif 'deberta' in args.backbone:
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/' + args.backbone, use_fast=True)
        model = DebertaForSequenceClassification.from_pretrained('microsoft/' + args.backbone, config=config)
    elif 'xlnet' in args.backbone:
        tokenizer = XLNetTokenizer.from_pretrained(args.backbone, use_fast=True)
        model = XLNetForSequenceClassification.from_pretrained(args.backbone, config=config)
    elif 'gpt' in args.backbone:
        tokenizer = GPT2Tokenizer.from_pretrained(args.backbone, use_fast=True)
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2ForSequenceClassification.from_pretrained(args.backbone, config=config)
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id
    elif 'electra' in args.backbone:
        tokenizer = ElectraTokenizer.from_pretrained('google/' + args.backbone, use_fast=True)
        model = ElectraForSequenceClassification.from_pretrained('google/' + args.backbone, config=config)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.backbone, use_fast=True)
        model = RobertaForSequenceClassification.from_pretrained(args.backbone, config=config)
    tokenizer.name = args.backbone

    return tokenizer, model.cuda()