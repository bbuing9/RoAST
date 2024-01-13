# Modified from: https://github.com/PKUnlp-icler/ChildTuning
import random
import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Tuple
from torch.distributions.bernoulli import Bernoulli
import math
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup
from training.common import cut_input
from tqdm import tqdm
import numpy as np
import json
import pickle

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

def calculate_fisher(args, model, loader):
    '''
    Calculate Fisher Information for different parameters
    '''
    gradient_mask = dict()
    model.train()

    N = len(loader)
    
    print('Gradient Mask')
    criterion = torch.nn.CrossEntropyLoss()

    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.cuda()
        labels = labels.cuda()

        _, logits, _ = model(input_ids=tokens, labels=labels, ours=True)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        if i == 0:
            for name, params in model.named_parameters():
                if 'layer' in name or 'transformer.h' in name:
                    if params.grad is not None:
                        gradient_mask[params] = params.new_zeros(params.size())  
        else: 
            for name, params in model.named_parameters():
                if 'layer' in name or 'transformer.h' in name:
                    if params.grad is not None:
                        gradient_mask[params] += (params.grad ** 2) / N
        model.zero_grad()

    print('Calculate Fisher Information')

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
            raise ValueError('error')
        
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
    print('===== Soft Masking with Sparsity {} ====='.format(all_sum / n_params))

    return gradient_mask

def set_optimizer(args, model, loader, total_steps):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (0.9, 0.98),
        "eps": 1e-6,
        "lr": args.model_lr,
    }

    if args.alpha > 0 and args.alpha < 1:
        gradient_mask = calculate_fisher(args, model, loader)
        optimizer = ChildTuningAdamW_online(optimizer_grouped_parameters, **optimizer_kwargs)
        optimizer.set_gradient_mask(gradient_mask)
    else:    
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    
    return optimizer, scheduler

class ChildTuningAdamW_online(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p = 1.0,
        mode = 'ChildTuning-D'
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.gradient_mask_save = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        flag = False
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================         
                if p in self.gradient_mask:
                    grad *= self.gradient_mask[p]
                    temp = self.gradient_mask_save[p].clone()
                    del self.gradient_mask_save[p]
                    flag = True
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                if flag:
                    self.gradient_mask_save[p] = temp
                    flag = False
        
        return loss
