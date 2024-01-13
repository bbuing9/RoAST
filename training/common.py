import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math

import operator
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def one_hot(labels, n_classes):
    one_hot = torch.zeros(labels.size(0), n_classes).to(device)
    one_hot[torch.arange(labels.size(0)), labels] = 1
    return one_hot

def uniform_labels(labels, n_classes):
    unif = torch.ones(labels.size(0), n_classes).to(device)
    return unif / n_classes

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
        return tokens[:, n_tokens-max_len:], attention_mask[:, n_tokens-max_len:]
    else:
        return tokens[:, :max_len], attention_mask[:, :max_len]

def sym_kld(aug_logits, orig_logits):
    return (F.kl_div(F.log_softmax(aug_logits, dim=-1, dtype=torch.float32),
                       F.softmax(orig_logits, dim=-1, dtype=torch.float32),
                       None, None,"none",)\
           + F.kl_div(F.log_softmax(orig_logits, dim=-1, dtype=torch.float32),
               F.softmax(aug_logits, dim=-1, dtype=torch.float32),
               None, None, "none",)
           ).sum(dim=1)
