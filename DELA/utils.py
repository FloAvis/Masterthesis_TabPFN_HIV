import torch
import numpy as np
import random
import os
import shutil

def init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def generate_default_config():
    configs = {}
    
    # Training parameters
    configs['train_batch_size'] = 128
    configs['shuffle'] = True
    configs['data_standardizing'] = False
    configs['lr'] = 1e-3
    configs['weight_decay'] = 1e-4
    configs['start_epoch'] = 0
    configs['max_epoch'] = 200
    configs['drop_ratio'] = 0.0
    configs['off_noise'] = 0.5
    configs['tau'] = 2/3
    configs['lr_ratio'] = 1.0
    configs['lr_scheduler'] = "none"
    configs['scheduler_warmup_epoch'] = 0
    configs['scheduler_decay_epoch'] = 0
    configs['scheduler_decay_rate'] = 0.5
    
    # Training information display and log
    configs['display_freq'] = 10 # display per 10 iterations
    configs['save_checkpoint_path'] = 'checkpoint'
    
    # Reproducibility
    configs['rand_seed'] = 0
    
    # Testing parameters
    configs['test_batch_size'] = 2 * configs['train_batch_size']
    configs['label_metrics'] = ["HammingLoss"]
    configs['score_metrics'] = ["OneError", "Coverage", "RankingLoss",
                                "AveragePrecision", "MacroAUC"]
    configs['number_metrics'] = len(configs['label_metrics']) + len(configs['score_metrics'])
    
    # Others
    configs['dtype'] = torch.float
    configs['eps'] = 1e-5
    
    return configs

def clear_old_logs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)