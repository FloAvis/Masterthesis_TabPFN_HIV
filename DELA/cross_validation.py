import torchnet
import os
import torch

def cross_validation(model, dataset, nfold=10, shuffle=True, random_state=None,
                     eval_on_trainset=False, quiet_mode=True, save_model=False):
    '''
    Evaluating model by cross-validation and select the best model for each round.

    Parameters
    ----------
    model :
        An instance of a learning model waiting for cross-validation, which
        should has 'train', 'test' and 'reset_parameters' functions realized.
    dataset : 
        An instance of dataset on which cross-validation is conducted.
    nfold : int, optional
        Number of folds for cross-validation. The default is 10.
    shuffle : bool, optional
        If shuffle=True, shuffling the data before cross-validation.
        The default is False.
    random_state : int, optional
        When shuffle is True, random_state affects the ordering of the indices,
        which controls the randomness of each fold. Otherwise, this parameter
        has no effect. The default is None.
    eval_on_trainset: bool, optional
        If eval_on_trainset=True, also evaluating on the training set.
        The default is False.
    quiet_mode : bool, optional
        If quiet_mode=True, training information will not be displayed during
        training.
        The default is True.
    save_model : bool, optional
        If save_model=True, save models for each fold.
        The default is False.
        
    Returns
    -------
    test_metrics : dict
        Metrics on the test set. Metrics returned are determined by model's
        'test' function.
    train_metrics: dict
        Metrics on the training set, only valid when eval_on_trainset=True. 
        Metrics returned are determined by model's 'test' function.
    '''
    # Creating metric recorder
    test_metrics = {}
    for metric in model.configs['label_metrics']:
        test_metrics[metric] = torchnet.meter.AverageValueMeter()
    for metric in model.configs['score_metrics']:
        test_metrics[metric] = torchnet.meter.AverageValueMeter()
        
    if eval_on_trainset:
        train_metrics = {}
        for metric in model.configs['label_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
        for metric in model.configs['score_metrics']:
            train_metrics[metric] = torchnet.meter.AverageValueMeter()
    else:
        train_metrics = None
    
    for count in range(1, nfold+1):
        print('Cross-validation: [{}/{}].'.format(count, nfold))
        
        dataset.data_cv_splitter(count, nfold, shuffle, random_state)
        train_dataloader = dataset.train_dataloader
        val_dataloader = dataset.val_dataloader
        test_dataloader = dataset.test_dataloader
        
        # Training with evaluation
        model.reset_parameters()
        model.configs['best_epoch'] = 0
        model.train(train_dataloader, val_dataloader, quiet_mode=quiet_mode)
        model.load_checkpoint(model.configs['best_checkpoint_path'])
        model.configs['start_epoch'] = 0

        # Testing
        metrics = model.test(test_dataloader)
        for key in metrics:
            test_metrics[key].add(metrics[key])
        
        if eval_on_trainset:
            metrics = model.Test(train_dataloader)
            for key in metrics:
                train_metrics[key].add(metrics[key])
        
        # Saving models
        if save_model:
            fileName = 'checkpoint_{:d}_{:d}_{:d}_cv'.format(shuffle, random_state, nfold)
            path = os.path.join(model.configs['save_checkpoint_path'],
                                fileName+'{:d}.pth'.format(count))
            _save_checkpoint({'epoch': model.configs['best_epoch'],
                              'state_dict': model.net.state_dict()}, path)
    
    return test_metrics, train_metrics

def _save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)
    