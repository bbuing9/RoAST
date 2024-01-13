import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, average_precision_score
import scipy.stats as stats
from utils import Logger, set_seed, save_model

from training.common import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_func(args, model, val_loader, test_loader, logger, log_dir, dataset, best_acc, final_acc, str=None):
    # other_metric; [mcc, f1, p, s]
    acc, other_metric, _ = test_acc(args, val_loader, model, logger)

    if args.dataset == 'cola':
        metric = other_metric[0]
    elif args.dataset == 'stsb':
        metric = other_metric[2]
    else:
        metric = acc

    if metric >= best_acc:
        # As val_data == test_data in GLUE, do not inference it again.
        if args.dataset == 'wnli' or args.dataset == 'rte' or args.dataset == 'mrpc' or args.dataset == 'stsb' or \
                args.dataset == 'cola' or args.dataset == 'sst2' or args.dataset == 'qnli' or args.dataset == 'qqp' :
            t_acc, t_other_metric = acc, other_metric
        else:
            t_acc, t_other_metric, _ = test_acc(args, test_loader, model, logger)

        if args.dataset == 'cola':
            t_metric = t_other_metric[0]
        elif args.dataset == 'stsb':
            t_metric = t_other_metric[2]
        else:
            t_metric = t_acc

        # Update test accuracy based on validation performance
        best_acc = metric
        final_acc = t_metric

        if args.dataset == 'mrpc' or args.dataset == 'qqp':
            logger.log('========== Test Acc/F1 ==========')
            logger.log('Test acc: {:.3f} Test F1: {:.3f}'.format(final_acc, t_other_metric[1]))
        elif args.dataset == 'stsb':
            logger.log('========== Test P/S ==========')
            logger.log('Test P: {:.3f} Test S: {:.3f}'.format(t_other_metric[2], t_other_metric[3]))
        elif args.dataset == 'mnli':
            logger.log('========== Test m/mm ==========')
            logger.log('Test matched/mismatched: {:.3f}/{:.3f}'.format(best_acc, final_acc))
        else:
            logger.log('========== Val Acc ==========')
            logger.log('Val acc: {:.3f}'.format(best_acc))
            logger.log('========== Test Acc ==========')
            logger.log('Test acc: {:.3f}'.format(final_acc))

        # Save model
        logger.log('Save model...')
        save_model(args, model, log_dir, dataset, str)

    return best_acc, final_acc

def cut_input(args, tokens):
    n_tokens = tokens.shape[1]
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
    elif 'gpt' in args.backbone:
        attention_mask = (tokens != 50256).float()
    elif 'xlnet' in args.backbone:
        attention_mask = (tokens != 5).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    if 'gpt' in args.backbone or 'xlnet' in args.backbone:
        return tokens[:, n_tokens-max_len:]
    else:
        return tokens[:, :max_len]

def acc_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results

def test_acc(args, loader, model, logger=None, binary=False):
    if logger is not None:
        logger.log('Compute test accuracy...')
    model.eval()

    error_top1 = AverageMeter()
    all_logits = []
    all_labels = []
    all_indices = []

    for i, (tokens, labels, indices) in enumerate(loader):
        batch_size = tokens.size(0)
        tokens = cut_input(args, tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)
        if len(labels.shape) == 2:
            labels = labels[:, 0]
        indices = indices.to(device)

        with torch.no_grad():
            _, logits, _ = model(input_ids=tokens, labels=labels)
            #outputs = model(tokens)  # (B, C)

        top1, = acc_k(logits.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        all_logits.append(logits)
        all_labels.append(labels)
        all_indices.append(indices)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    if args.dataset != 'stsb':
        all_preds = all_logits.cpu().max(1)[1]
        if binary:
            all_preds[all_preds == 1] = 2 # Following the original author of WANLI, both neutral & contradict are considered as non-entailment
    else:
        all_preds = all_logits.cpu()
    all_labels = all_labels.cpu()
    acc = 100.0 * (all_preds == all_labels).float().sum() / len(all_logits)

    ece_avg = 0.0
    rand_idx = torch.randperm(len(all_logits))
    n_interval = int((1 / args.n_eval_ece) * len(all_logits))
    for j in range(args.n_eval_ece):
        val_logits, val_labels = all_logits[rand_idx][j * n_interval:(j+1) * n_interval].cpu(), all_labels[rand_idx][j * n_interval:(j+1) * n_interval] 
        # TODO: Simplification of the below parts (for code readability)
        if j == 0:
            test_logits = all_logits[rand_idx][(j+1) * n_interval:].cpu()
            test_labels = all_labels[rand_idx][(j+1) * n_interval:].cpu()
        elif j == (args.n_eval_ece - 1):
            test_logits = all_logits[rand_idx][:j * n_interval].cpu()
            test_labels = all_labels[rand_idx][:j * n_interval].cpu()
        else: 
            test_logits = torch.cat([all_logits[rand_idx][:j * n_interval], all_logits[rand_idx][(j+1) * n_interval:]], dim=0).cpu()
            test_labels = torch.cat([all_labels[rand_idx][:j * n_interval], all_labels[rand_idx][(j+1) * n_interval:]], dim=0).cpu()

        _, ece_j_temp = ECE(val_logits, val_labels)
        ece_j, _  = ECE(test_logits, test_labels, ece_j_temp)
        ece_avg += (ece_j / args.n_eval_ece)

    # Calculate the F1, MCC
    f1, mcc, p, s = 0, 0, 0, 0

    if args.dataset == 'cola':
        mcc = matthews_corrcoef(all_preds, all_labels)
    elif args.dataset == 'stsb':
        p = stats.pearsonr(all_labels, all_preds[:, 0])[0]
        s = stats.spearmanr(all_labels, all_preds[:, 0])[0]
    elif args.dataset == 'mrpc' or args.dataset == 'qqp':
        f1 = f1_score(all_labels, all_preds)

    return acc, [100 * mcc, 100 * f1, 100 * p, 100 * s], ece_avg

def ECE(logits, labels, opt_temp=None):
    preds = torch.softmax(logits, 1)
    sorted_pred, sorted_idx = torch.sort(preds, dim=1, descending=True)

    top_i_acc = 0
    top_i_confidence = 0
    ece_list = []
    if opt_temp is None:
        temps = [0.25, 0.5, 1, 2, 4, 8]
    else:
        temps = [opt_temp]
    for temp in temps:
        preds = torch.softmax(logits / temp, 1)
        sorted_pred, sorted_idx = torch.sort(preds, dim=1, descending=True)
        i_th_pred = sorted_pred[:,0]
        top_i_confidence = i_th_pred
        i_th_correct = sorted_idx[:,0].eq(labels.data).float()
        top_i_acc = i_th_correct

        n_bins = 10

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = top_i_confidence.gt(bin_lower.item()) * top_i_confidence.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = top_i_acc[in_bin].float().mean()
                avg_confidence_in_bin = top_i_confidence[in_bin].mean()
                ece += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item()
        ece_list.append(ece)

    best_ece = min(ece_list)
    best_temp = np.argmin(np.array(ece_list))

    return best_ece, best_temp

def get_ood_metrics(args, ind_loader, ood_loader, model, binary=False):
    model.eval()

    d_in, d_ood = [], []
    for i, (tokens, labels, indices) in enumerate(ind_loader):
        batch_size = tokens.size(0)
        tokens = cut_input(args, tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)
        if binary:
            labels[labels == 2] = 1
        if len(labels.shape) == 2:
            labels = labels[:, 0]
        indices = indices.to(device)

        with torch.no_grad():
            _, logits, _ = model(input_ids=tokens, labels=labels)
        probs = logits.softmax(dim=-1)
        msp = probs.max(dim=-1)[0]
        d_in.append(msp)

    d_in = -1 * torch.cat(d_in, dim=0).cpu().numpy()

    for i, (tokens, labels, indices) in enumerate(ood_loader):
        batch_size = tokens.size(0)
        tokens = cut_input(args, tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)
        indices = indices.to(device)

        if binary:
            labels[labels == 2] = 1
        if len(labels.shape) == 2:
            labels = labels[:, 0]

        with torch.no_grad():
            _, logits, _ = model(input_ids=tokens, labels=labels)
        probs = logits.softmax(dim=-1)
        msp = probs.max(dim=-1)[0]
        d_ood.append(msp)
    d_ood = -1 * torch.cat(d_ood, dim=0).cpu().numpy()

    fpr95 = get_fpr(d_in, d_ood)
    auroc, aupr = get_roc_sklearn(d_in, d_ood), get_pr_sklearn(d_in, d_ood)
    return fpr95, auroc, aupr

def get_fpr(xin, xood):
    return np.sum(np.percentile(xood, 5) < xin) / len(xin) 

def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = roc_auc_score(labels, data)
    return auroc

def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = average_precision_score(labels, data)
    return aupr
