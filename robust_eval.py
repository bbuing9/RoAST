import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from evals import eval_func, test_acc, get_ood_metrics
from data import get_base_dataset
from models import load_models
from training import train_base
from common import CKPT_PATH, parse_args
from utils import Logger, set_seed, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def call_eval(args, model, log_name, logger):
    # Set seed
    set_seed(args)

    # Set logs
    log_name = './evals/' + log_name
    log_dir = logger.logdir

    logger.log('Loading the model with best eval performance')
    logger.log(args.pre_ckpt)
    model.load_state_dict(torch.load(log_name + "/model"))

    if args.dataset == 'mnli' or args.dataset == 'diagnostics' or args.dataset == 'fever-nli' or args.dataset == 'wanli':
        args.n_class = 3
    else:
        args.n_class = 2
    tokenizer, _ = load_models(args)
    
    logger.log('========== 1. Start evaluation on multiple distribution shifted datasets... ==========')
    if args.dataset == 'mnli':
        eval_datasets = ["mnli_m", "mnli_mm", "diagnostics", "hans", "qnli", "wnli", "nq-nli", "fever-nli", "wanli"]
    else:
        eval_datasets = ["sst2", "yelp", "imdb", "cimdb", "poem", "amazon"]

    all_res = []
    all_ece = []
    for eval_data in eval_datasets:
        logger.log('==========> Start evaluation ({})'.format(eval_data))
        if args.dataset == 'mnli':
            eval_dataset = get_base_dataset(eval_data, tokenizer, args.seed, eval=True)
        else:
            eval_dataset = get_base_dataset(eval_data, tokenizer, args.seed, sent=True)
        
        eval_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
        
        if eval_data == "hans" or eval_data == "qnli" or eval_data == "wnli" or eval_data == "nq-nli":
            binary = True
        else:
            binary = False

        acc, _, ece = test_acc(args, eval_loader, model, logger, binary)
        logger.log('Test accuracy: {}, ece: {}'.format(acc, ece))
        all_res.append(int(1000 * float(acc.data)) / 1000)
        all_ece.append(int(1000 * ece) / 1000)
    
    logger.log('========== 2. Start evaluation on adversarially perturbed texts (black-box setup)... ==========')
    if args.dataset == 'mnli':
        adv_datasets = ["infobert_mnli_matched", "infobert_mnli_mismatched", "roberta_mnli_matched", "roberta_mnli_mismatched", "anli1", "anli2", "anli3", "advglue_mnli_m", "advglue_mnli_mm"]
    else:
        adv_datasets = ["adv_bert_sst2", "adv_roberta_sst2", "dynasent1", "dynasent2", "advglue_sst2"]
    
    all_adv_res = []
    for adv_data in adv_datasets:
        logger.log('==========> Start adversarial evaluation ({})'.format(adv_data))
        if args.dataset == 'mnli' or adv_data == "advglue_sst2":
            if 'anli' in adv_data:
                eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, eval=True)
            else:
                eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, adv=True)
        else:
            eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, sent=True)
        
        eval_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

        acc, _, ece = test_acc(args, eval_loader, model, logger, False)
        logger.log('Test accuracy: {}, ece: {}'.format(acc, ece))
        all_adv_res.append(int(1000 * float(acc.data)) / 1000)    
        all_ece.append(int(1000 * ece) / 1000)   

    logger.log('========== 3. Start evaluation on abnormal detection (black-box setup)... ==========')
    ind_dataset = get_base_dataset(args.dataset, tokenizer, args.seed)
    ind_loader = DataLoader(ind_dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    
    if args.dataset == 'mnli':
        ood_datasets = ["wmt16", "multi30k", "sst2", "20news", "qqp"]
    else:
        ood_datasets = ["wmt16", "multi30k", "20news", "qqp", "mnli_m", "mnli_mm"] 
    
    all_ood_auroc = []
    all_ood_aupr = []
    all_ood_fpr = []
    for ood_data in ood_datasets:
        logger.log('==========> Start OOD detection evaluation ({})'.format(ood_data))
        eval_dataset = get_base_dataset(ood_data, tokenizer, args.seed, ood=True)
        ood_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

        if args.dataset == 'mnli':
            fpr95, auroc, aupr = get_ood_metrics(args, ind_loader, ood_loader, model)
        else:
            fpr95, auroc, aupr = get_ood_metrics(args, ind_loader, ood_loader, model, binary=True)
        logger.log('AUROC: {}, AUPR: {}, FPR95: {}'.format(auroc, aupr, fpr95))
        all_ood_auroc.append(int(1000 * auroc) / 1000)   
        all_ood_aupr.append(int(1000 * aupr) / 1000)
        all_ood_fpr.append(int(1000 * fpr95) / 1000) 
    
    logger.log('All Evaluation Results')
    
    logger.log(eval_datasets)
    logger.log(all_res)
    np.save(log_dir + '/all_res.npy', all_res)
    
    logger.log(adv_datasets)
    logger.log(all_adv_res)
    np.save(log_dir + '/all_adv_res.npy', all_adv_res)

    logger.log(ood_datasets)
    logger.log(all_ood_auroc)
    np.save(log_dir + '/all_ood_auroc.npy', all_ood_auroc)
    logger.log(all_ood_aupr)
    np.save(log_dir + '/all_ood_aupr.npy', all_ood_aupr)
    logger.log(all_ood_fpr)
    np.save(log_dir + '/all_ood_fpr.npy', all_ood_fpr)
    
    logger.log(all_ece)
    np.save(log_dir + '/all_ece.npy', all_ece)
    
    if args.dataset == 'mnli':
        logger.log('inD: {}, ooD: {}, adv: {}, ece: {}, auroc: {}'.format(sum(all_res[:2]) / len(all_res[:2]),
                                                                          sum(all_res[2:]) / len(all_res[2:]), 
                                                                          sum(all_adv_res) / len(all_adv_res),
                                                                          sum(all_ece) / len(all_ece),
                                                                          sum(all_ood_auroc) / len(all_ood_auroc)))
    else:    
        logger.log('inD: {}, ooD: {}, adv: {}, ece: {}, auroc: {}'.format(all_res[0], 
                                                                          sum(all_res[1:]) / len(all_res[1:]), 
                                                                          sum(all_adv_res) / len(all_adv_res),
                                                                          sum(all_ece) / len(all_ece),
                                                                          sum(all_ood_auroc) / len(all_ood_auroc)))

    logger.log("===== Write CSV file (loc: {})=====".format(log_name + '.csv'))
    with open(log_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        if args.dataset == 'mnli':
            ind_list = [sum(all_res[:2]) / 2, all_res[0], all_res[1]]
            ood_list = [sum(all_res[2:]) / len(all_res[2:])] + all_res[2:]
        else:
            ind_list = [all_res[0]]
            ood_list = [sum(all_res[1:]) / len(all_res[1:])] + all_res[1:]
        adv_list = [sum(all_adv_res) / len(all_adv_res)] + all_adv_res
        auroc_list = [sum(all_ood_auroc) / len(all_ood_auroc)] + all_ood_auroc
        aupr_list = [sum(all_ood_aupr) / len(all_ood_aupr)] + all_ood_aupr
        fpr_list = [sum(all_ood_fpr) / len(all_ood_fpr)] + all_ood_fpr
        ece_list = [sum(all_ece) / len(all_ece)] + all_ece
        
        writer.writerow(ind_list + ood_list + adv_list + auroc_list + aupr_list + fpr_list + ece_list)

def main():
    args = parse_args(mode='train')

    # Set seed
    set_seed(args)

    # Set logs
    log_name = f"{args.task}_{args.eval_type}"
    logger = Logger(log_name)
    log_dir = logger.logdir

    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    if args.task == 'entailment':
        args.dataset = 'mnli'
        args.n_class = 3
    elif args.task == 'sentiment':
        args.dataset = 'sst2'
        args.n_class = 2
    else:
        raise ValueError('Wrong Task')
    tokenizer, model = load_models(args)

    logger.log('Loading from pre-trained model')
    logger.log(args.pre_ckpt)
    model.load_state_dict(torch.load(args.pre_ckpt))

    logger.log('========== 1. Start evaluation on multiple distribution shifted datasets... ==========')
    if args.dataset == 'mnli':
        eval_datasets = ["mnli_m", "mnli_mm", "diagnostics", "hans", "qnli", "wnli", "nq-nli", "fever-nli", "wanli"]
    else:
        eval_datasets = ["sst2", "yelp", "imdb", "cimdb", "poem", "amazon"]
    
    all_res = []
    all_ece = []
    for eval_data in eval_datasets:
        logger.log('==========> Start evaluation ({})'.format(eval_data))
        if args.dataset == 'mnli':
            eval_dataset = get_base_dataset(eval_data, tokenizer, args.seed, eval=True)
        else:
            eval_dataset = get_base_dataset(eval_data, tokenizer, args.seed, sent=True)
        
        eval_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
        
        if eval_data == "hans" or eval_data == "qnli" or eval_data == "wnli" or eval_data == "nq-nli":
            binary = True
        else:
            binary = False

        acc, _, ece = test_acc(args, eval_loader, model, logger, binary)
        logger.log('Test accuracy: {}, ece: {}'.format(acc, ece))
        all_res.append(int(1000 * float(acc.data)) / 1000)
        all_ece.append(int(1000 * ece) / 1000)
    
    logger.log('========== 2. Start evaluation on adversarially perturbed texts (black-box setup)... ==========')
    if args.dataset == 'mnli':
        adv_datasets = ["infobert_mnli_matched", "infobert_mnli_mismatched", "roberta_mnli_matched", "roberta_mnli_mismatched", "anli1", "anli2", "anli3", "advglue_mnli_m", "advglue_mnli_mm"]
    else:
        adv_datasets = ["adv_bert_sst2", "adv_roberta_sst2", "dynasent1", "dynasent2", "advglue_sst2"]
    
    all_adv_res = []
    for adv_data in adv_datasets:
        logger.log('==========> Start adversarial evaluation ({})'.format(adv_data))
        if args.dataset == 'mnli' or adv_data == "advglue_sst2":
            if 'anli' in adv_data:
                eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, eval=True)
            else:
                eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, adv=True)
        else:
            eval_dataset = get_base_dataset(adv_data, tokenizer, args.seed, sent=True)
        
        eval_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

        acc, _, ece = test_acc(args, eval_loader, model, logger, False)
        logger.log('Test accuracy: {}, ece: {}'.format(acc, ece))
        all_adv_res.append(int(1000 * float(acc.data)) / 1000)    
        all_ece.append(int(1000 * ece) / 1000)   

    logger.log('========== 3. Start evaluation on abnormal detection (black-box setup)... ==========')
    ind_dataset = get_base_dataset(args.dataset, tokenizer, args.seed)
    ind_loader = DataLoader(ind_dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    
    if args.dataset == 'mnli':
        ood_datasets = ["wmt16", "multi30k", "sst2", "20news", "qqp"]
    else:
        ood_datasets = ["wmt16", "multi30k", "20news", "qqp", "mnli_m", "mnli_mm"] 
    
    all_ood_auroc = []
    all_ood_aupr = []
    all_ood_fpr = []
    for ood_data in ood_datasets:
        logger.log('==========> Start OOD detection evaluation ({})'.format(ood_data))
        if 'mnli' in ood_data:
            eval_dataset = get_base_dataset(ood_data, tokenizer, args.seed, eval=True)
        elif 'sst2' in ood_data:
            eval_dataset = get_base_dataset(ood_data, tokenizer, args.seed, sent=True)
        else:
            eval_dataset = get_base_dataset(ood_data, tokenizer, args.seed, ood=True)
        ood_loader = DataLoader(eval_dataset.eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

        if args.dataset == 'mnli':
            fpr95, auroc, aupr = get_ood_metrics(args, ind_loader, ood_loader, model)
        else:
            fpr95, auroc, aupr = get_ood_metrics(args, ind_loader, ood_loader, model, binary=True)
        logger.log('AUROC: {}, AUPR: {}, FPR95: {}'.format(auroc, aupr, fpr95))
        all_ood_auroc.append(int(1000 * auroc) / 1000)   
        all_ood_aupr.append(int(1000 * aupr) / 1000)
        all_ood_fpr.append(int(1000 * fpr95) / 1000) 
    
    logger.log('All Evaluation Results')
    
    logger.log(eval_datasets)
    logger.log(all_res)
    np.save(log_dir + '/all_res.npy', all_res)
    
    logger.log(adv_datasets)
    logger.log(all_adv_res)
    np.save(log_dir + '/all_adv_res.npy', all_adv_res)

    logger.log(ood_datasets)
    logger.log(all_ood_auroc)
    np.save(log_dir + '/all_ood_auroc.npy', all_ood_auroc)
    logger.log(all_ood_aupr)
    np.save(log_dir + '/all_ood_aupr.npy', all_ood_aupr)
    logger.log(all_ood_fpr)
    np.save(log_dir + '/all_ood_fpr.npy', all_ood_fpr)
    
    logger.log(all_ece)
    np.save(log_dir + '/all_ece.npy', all_ece)
    
    if args.dataset == 'mnli':
        logger.log('inD: {}, ooD: {}, adv: {}, ece: {}, auroc: {}'.format(sum(all_res[:2]) / len(all_res[:2]),
                                                                          sum(all_res[2:]) / len(all_res[2:]), 
                                                                          sum(all_adv_res) / len(all_adv_res),
                                                                          sum(all_ece) / len(all_ece),
                                                                          sum(all_ood_auroc) / len(all_ood_auroc)))
    else:    
        logger.log('inD: {}, ooD: {}, adv: {}, ece: {}, auroc: {}'.format(all_res[0], 
                                                                          sum(all_res[1:]) / len(all_res[1:]), 
                                                                          sum(all_adv_res) / len(all_adv_res),
                                                                          sum(all_ece) / len(all_ece),
                                                                          sum(all_ood_auroc) / len(all_ood_auroc)))

    logger.log("===== Write CSV file (loc: {})=====".format(log_name + '.csv'))
    with open(log_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        if args.dataset == 'mnli':
            ind_list = [sum(all_res[:2]) / 2, all_res[0], all_res[1]]
            ood_list = [sum(all_res[2:]) / len(all_res[2:])] + all_res[2:]
        else:
            ind_list = [all_res[0]]
            ood_list = [sum(all_res[1:]) / len(all_res[1:])] + all_res[1:]
        adv_list = [sum(all_adv_res) / len(all_adv_res)] + all_adv_res
        auroc_list = [sum(all_ood_auroc) / len(all_ood_auroc)] + all_ood_auroc
        aupr_list = [sum(all_ood_aupr) / len(all_ood_aupr)] + all_ood_aupr
        fpr_list = [sum(all_ood_fpr) / len(all_ood_fpr)] + all_ood_fpr
        ece_list = [sum(all_ece) / len(all_ece)] + all_ece
        
        writer.writerow(ind_list + ood_list + adv_list + auroc_list + aupr_list + fpr_list + ece_list)
if __name__ == "__main__":
    main()
