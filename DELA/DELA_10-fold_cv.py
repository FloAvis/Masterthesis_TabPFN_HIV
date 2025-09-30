import sys
import os
import argparse
import torch
import yaml

sys.path.append("./DELA")
from DELAModel import DELAModel
from dataset import *
from cross_validation import cross_validation
from utils import init_random_seed, generate_default_config, clear_old_logs

parser = argparse.ArgumentParser()
parser.add_argument('exp',
                    help='name of experiment')
parser.add_argument('--dataset', '-dataset', type=str, default="corel5k",
                    help='dataset on which experiment is conducted')
parser.add_argument('--batch_size', '-bs', type=int, default=128,
                    help='batch size for one iteration during training')
parser.add_argument('--lr', '-lr', type=float, default=1e-3,
                    help='learning rate parameter')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4,
                    help='weight decay parameter')
parser.add_argument('--dropout', '-dropout', type=float, default=0.0,
                    help='dropout probability')
parser.add_argument('--max_epoch', '-max_epoch', type=int, default=200,
                    help='maximal training epochs')
parser.add_argument('--latent_dim', '-latent_dim', type=int, default=50,
                    help='dimensionality of latent representation')
parser.add_argument('--beta', '-beta', type=float, default=1.0,
                    help='trade-off parameter')
parser.add_argument('--lr_scheduler', '-lr_s', type=str, default="none",
                    choices=["none", "step_epoch"],
                    help='learning rate schedule used during training')
parser.add_argument('--step', '-step', type=int, default=0,
                    help='period of learning rate decay (in epoch)')
parser.add_argument('--warmup', '-warmup', type=int, default=0,
                    help='warmup epochs for learning rate')
parser.add_argument('--cuda', '-cuda', action='store_true',
                    help='whether to use gpu')
parser.add_argument('--quiet', '-quiet', action='store_true',
                    help='whether to train in quiet mode')
parser.add_argument('--default_cfg', '-default_cfg', action='store_true',
                    help='whether to run experiment with default hyperparameters')

binary_datasets = ["corel5k", "Corel16k001", "delicious", "tmc2007", "bookmarks"]

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Setting random seeds
    init_random_seed()
    
    # Setting configurations
    configs = generate_default_config()
    # device params
    configs['use_gpu'] = args.cuda
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
    # training params
    configs['train_batch_size'] = args.batch_size
    configs['test_batch_size'] = 2 * configs['train_batch_size']
    configs['max_epoch'] = args.max_epoch
    configs['lr'] = args.lr
    configs['weight_decay'] = args.weight_decay
    configs['drop_ratio'] = args.dropout
    configs['beta'] = args.beta
    configs['lr_scheduler'] = args.lr_scheduler
    configs['scheduler_warmup_epoch'] = args.warmup
    configs['scheduler_decay_epoch'] = args.step

    # Loading dataset
    configs['shuffle'] = True
    if args.dataset in binary_datasets:
        configs['data_standardizing'] = False
    else:
        configs['data_standardizing'] = True
    print(args.dataset)
    dataset = eval(args.dataset)(configs=configs)
    configs['dataset_name'] = dataset.name()
    
    # Setting architecture params
    configs['model_name'] = 'DELAModel'
    configs['in_features'] = dataset.feat_dim
    configs['num_classes'] = dataset.num_class
    configs['latent_dim'] = args.latent_dim
    
    # Setting other params
    configs['exp'] = args.exp
    configs['exp_dir'] = os.path.join(configs['model_name'],
                                      configs['exp'],
                                      configs['dataset_name'])
    configs['save_checkpoint_path'] = os.path.join(configs['exp_dir'], 'checkpoint')
    
    # Loading dataset-specific configs
    if args.default_cfg:
        cfgs_file = os.path.join('./DELA/configs', configs['dataset_name']+'_cfgs.yaml')
        if os.path.exists(cfgs_file):
            print('Loading configs from {}'.format(cfgs_file))
            with open(cfgs_file, 'r') as f:
                cfgs = yaml.safe_load(f)
            print(cfgs)
            for key in cfgs.keys():
                configs[key] = cfgs[key]
        else:
            print('No config file is found in path {}'.format(cfgs_file))
    
    # Clearing old logs
    clear_old_logs(os.path.join(configs['model_name'], configs['exp']))
    
    # Creating model
    model = DELAModel(configs)
    
    # Cross-validation
    val_metrics, _ = cross_validation(model, dataset, random_state=configs['rand_seed'],
                                      quiet_mode=args.quiet, save_model=True)
    
    # Displaying results of cross-validation
    for key in val_metrics:
        print('{}: {:.4f} / {:.4f}'.format(key, val_metrics[key].value()[0], val_metrics[key].value()[1]))
    