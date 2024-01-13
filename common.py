import argparse

DATA_PATH = './dataset'
CKPT_PATH = './checkpoint'

def parse_args(mode):
    assert mode in ['train', 'eval']

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", help='tasks (entailment|sentiment)',
                        required=True, type=str)
    parser.add_argument("--data_ratio", help='imbalance ratio, i.e., |min class| / |max class|',
                        default=1.0, type=float)
    parser.add_argument("--backbone", help='backbone network',
                        default='roberta-large', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser = _parse_args_train(parser)

    return parser.parse_args()

def _parse_args_train(parser):
    # ========== Training ========== #
    parser.add_argument("--train_type", help='arguments to note the details of training',
                        default='base', type=str)
    parser.add_argument("--eval_type", help='arguments to note the details of evaluation',
                        default='base', type=str)                    
    parser.add_argument("--epochs", help='training epochs',
                        default=10, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=16, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--weight_decay", help='weight decay',
                        default=0.01, type=float)
    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--grad_accum", help='number of grad accumulation step',
                        default=1, type=int)
    parser.add_argument("--n_eval_ece", help='number of cross validation for ECE calculation',
                        default=10, type=int)            
    
    # ========== Multi-gpu Training ========== #
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--ngpu', type=int, default=4)

    # ========== RoAST ========== #
    parser.add_argument("--roast", help='applying roast for fine-tuning',
                        action='store_true')
                        
    parser.add_argument("--adv_eps", help='noise magnitude for adversarial training',
                        default=0.0, type=float)
    parser.add_argument("--coeff_sym", help='coefficient for symmetric KL divergence loss',
                        default=0.0, type=float)   

    parser.add_argument("--unbiased_scale", help='project sam',
                        action='store_true')
    parser.add_argument("--alpha", help='hyper-parameter to control the sparsity of masked gradient',
                        default=0.0, type=float)      
    parser.add_argument("--beta", help='hyper-parameter to control the smoothness of approximation',
                        default=0.0, type=float)
    
    return parser

