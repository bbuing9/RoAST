import os
import sys
import time
from datetime import datetime
import shutil
import math

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def generate_noise(embedding_output, epsilon=1e-5):
    noise = embedding_output.data.new(embedding_output.size()).uniform_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def norm_grad(grad, epsilon=1e-8, norm_p='l_inf'):
    if norm_p == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + epsilon)
    elif norm_p == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
    return direction

# Basic setups
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn, local_rank=None):
        self.local_rank = local_rank

        if self.local_rank is None or self.local_rank in [-1, 0]:    
            if not os.path.exists("./logs/"):
                os.mkdir("./logs/")

            logdir = 'logs/' + fn
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            if len(os.listdir(logdir)) != 0:
                ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                                "Will you proceed [y/N]? ")
                if ans in ['y', 'Y']:
                    shutil.rmtree(logdir)
                else:
                    exit(1)
            self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string, local_rank=None):
        if self.local_rank is None or self.local_rank in [-1, 0]:
            self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
            self.log_file.flush()

            print('[%s] %s' % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)

    return dataset.base_path + suffix + '.model'

def save_model(args, model, log_dir, dataset, str=None):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    model_path = set_model_path(args, dataset)
    if str is not None:
        save_path = os.path.join(log_dir, 'model' + str)
    else:    
        save_path = os.path.join(log_dir, 'model')
    torch.save(model.state_dict(), save_path)