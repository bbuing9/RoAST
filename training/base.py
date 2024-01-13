import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle

from training.common import AverageMeter, cut_input
from utils import get_parameter_names, generate_noise, norm_grad
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_base(args, loader, model, optimizer, scheduler, epoch=0, logger=None, pre_weights=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['p_avg'] = AverageMeter()
    losses['q_avg'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)

        tokens, _ = cut_input(args, tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)

        loss, logits, outputs = model(input_ids=tokens, labels=labels)
        
        p = torch.log_softmax(logits, dim=-1).data
        p_tec = torch.softmax(logits, dim=-1).data
                
        q_tec = p_tec
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # cls_acc
        orig_conf = p_tec[torch.arange(batch_size), labels].mean()
        adv_conf = q_tec[torch.arange(batch_size), labels].mean()

        _, pred_cls = logits.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['p_avg'].update(orig_conf.item(), batch_size)
        losses['q_avg'].update(adv_conf.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f] [OrigP %.3f] [AdvP %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average, losses['p_avg'].average, losses['q_avg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_roast(args, loader, model, optimizer, scheduler, epoch=0, logger=None, pre_weights=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['p_avg'] = AverageMeter()
    losses['q_avg'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')
    gradient_mask_save = dict()
    
    optimizer.gradient_mask_save = gradient_mask_save
    N = len(loader)

    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)

        loss, logits, outputs = model(input_ids=tokens, labels=labels)
        
        p = torch.log_softmax(logits, dim=-1).data
        p_tec = torch.softmax(logits, dim=-1).data        
        q_tec = p_tec
        loss.backward()

        if i == 0:
            for name, params in model.named_parameters():
                if 'layer' in name or 'transformer.h' in name:
                    if params.grad is not None:
                        gradient_mask_save[params] = params.new_zeros(params.size())
        else:
            for name, params in model.named_parameters():
                if 'layer' in name or 'transformer.h' in name:
                    if params.grad is not None:
                        optimizer.gradient_mask_save[params] += (params.grad ** 2) / N
        
        if (i + 1) % args.grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        orig_conf = p_tec[torch.arange(batch_size), labels].mean()
        adv_conf = q_tec[torch.arange(batch_size), labels].mean()

        _, pred_cls = logits.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['p_avg'].update(orig_conf.item(), batch_size)
        losses['q_avg'].update(adv_conf.item(), batch_size)

    gradient_mask = optimizer.gradient_mask_save
    
    # Numpy
    dict_shape = dict()
    r = None
    for k, v in gradient_mask.items():
        dict_shape[k] = v.shape
    
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    
    relative_order_prev = np.argsort(r)
    relative_order = np.argsort(relative_order_prev)

    n_params = len(relative_order) 
    relative_mask = relative_order / (n_params - 1)
    order_masks = dict()

    n_total = 0
    for k in gradient_mask:
        if len(dict_shape[k]) == 1:
            n_entries = dict_shape[k][0]
            k_order = relative_mask[n_total:n_total+n_entries]
        elif len(dict_shape[k]) == 2:
            n_entries = dict_shape[k][0] * dict_shape[k][1] 
            k_order = relative_mask[n_total:n_total+n_entries].reshape(dict_shape[k][0], dict_shape[k][1])
        elif len(dict_shape[k]) == 3:
            n_entries = dict_shape[k][0] * dict_shape[k][1] * dict_shape[k][2]
            k_order = relative_mask[n_total:n_total+n_entries].reshape(dict_shape[k][0], dict_shape[k][1], dict_shape[k][2])
        else:
            logger.log('error')
        
        order_masks[k] = k_order
        n_total += n_entries

    all_sum = 0
    for k in gradient_mask:
        soft_mask_k = 1 / (1 + torch.exp(-2 * args.beta * (torch.Tensor(order_masks[k]).cuda() - args.alpha))) 
        if args.unbiased_scale:
            gradient_mask[k] = (1/(soft_mask_k + 1e-12)) * torch.bernoulli(soft_mask_k)
        else:
            gradient_mask[k] = torch.bernoulli(soft_mask_k)
        all_sum += (gradient_mask[k] > 0).sum()
    logger.log('===== Soft Masking with Sparsity {} ====='.format(all_sum / n_params))
    optimizer.set_gradient_mask(gradient_mask)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f] [OrigP %.3f] [AdvP %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average, losses['p_avg'].average, losses['q_avg'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)