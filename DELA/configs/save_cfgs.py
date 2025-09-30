import yaml

def save_cfg_corel5k():
    configs = {'max_epoch': 50,
               'lr': 3e-3,
               'beta': 1e-4,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 4}
    
    return configs

def save_cfg_rcv1subset1():
    configs = {'max_epoch': 50,
               'lr': 3e-3,
               'beta': 1e-3,
               'drop_ratio': 0.7,
               'lr_scheduler': "step_epoch",
               'scheduler_warmup_epoch': 2,
               'scheduler_decay_epoch': 3}
    
    return configs

def save_cfg_Corel16k001():
    configs = {'max_epoch': 50,
               'lr': 1e-3,
               'beta': 1e-3,
               'drop_ratio': 0.1,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 4}
    
    return configs

def save_cfg_delicious():
    configs = {'max_epoch': 200,
               'lr': 1e-4,
               'beta': 1.0}
    
    return configs

def save_cfg_iaprtc12():
    configs = {'max_epoch': 50,
               'lr': 3e-3,
               'beta': 1e-1,
               'latent_dim': 128,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 8}
    
    return configs

def save_cfg_espgame():
    configs = {'max_epoch': 50,
               'lr': 3e-3,
               'beta': 1.0,
               'lr_ratio': 10.0,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 4}
    
    return configs

def save_cfg_mirflickr():
    configs = {'max_epoch': 50,
               'lr': 3e-3,
               'beta': 1e-2}
    
    return configs

def save_cfg_tmc2007():
    configs = {'max_epoch': 30,
               'lr': 3e-3,
               'beta': 1e-3,
               'lr_ratio': 0.1,
               'latent_dim': 64,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 1}
    
    return configs

def save_cfg_mediamill():
    configs = {'max_epoch': 250,
               'lr': 3e-4,
               'beta': 1e-2,
               'latent_dim': 256,
               'drop_ratio': 0.1,
               'lr_scheduler': "step_epoch",
               'scheduler_warmup_epoch': 40}

    return configs

def save_cfg_bookmarks():
    configs = {'max_epoch': 30,
               'lr': 1e-3,
               'beta': 1e-2,
               'lr_scheduler': "step_epoch",
               'scheduler_decay_epoch': 2}
    
    return configs

if __name__ == '__main__':
    datasets = ["corel5k", "rcv1subset1", "Corel16k001", "delicious", "iaprtc12",
                "espgame", "mirflickr", "tmc2007", "mediamill", "bookmarks"]
    for dataset in datasets: 
        filename = dataset + '_cfgs.yaml'
        func = 'save_cfg_' + dataset
        
        configs = eval(func)()
        
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(configs, f)