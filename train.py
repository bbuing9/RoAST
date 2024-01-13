import os

import torch
from torch.utils.data import DataLoader

from evals import eval_func
from data import get_base_dataset
from models import load_models
from training import train_base, train_roast
from common import CKPT_PATH, parse_args
from utils import Logger, set_seed, save_model
from roast_optim import set_optimizer, calculate_fisher
from robust_eval import call_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args(mode='train')

    # Set seed
    set_seed(args)

    # Set logs
    log_name = f"{args.task}_{args.train_type}_Alpha{args.alpha}_Beta{args.beta}_Adv{args.adv_eps}__Lsym{args.coeff_sym}_B{args.batch_size}_S{args.seed}"
    
    logger = Logger(log_name)
    log_dir = logger.logdir
    logger.log(args)

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

    logger.log('Initializing dataset...')
    dataset = get_base_dataset(args.dataset, tokenizer, args.seed)
    train_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(dataset.val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    
    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    total_steps = args.epochs * len(train_loader)
    optimizer, scheduler = set_optimizer(args, model, train_loader, total_steps)
    
    logger.log('Training model...')
    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc = 0, 0

    for epoch in range(1, args.epochs + 1):
        if args.roast:
            train_roast(args, train_loader, model, optimizer, scheduler, epoch, logger)
        else:
            train_base(args, train_loader, model, optimizer, scheduler, epoch, logger)
        best_acc, final_acc = eval_func(args, model, val_loader, test_loader, logger, log_dir, dataset, best_acc, final_acc)

    logger.log('===========>>>>> Final Test Accuracy: {}'.format(final_acc))

    logger.log('===========>>>>> Starting evaluation')
    call_eval(args, model, log_name, logger)

if __name__ == "__main__":
    main()
