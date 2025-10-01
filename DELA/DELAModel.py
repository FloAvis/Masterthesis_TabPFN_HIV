import torch
import math
import os
import numpy as np

from DELA.DELANet import DELANet
from DELA.optims.Engine import DELAModelEngine
from DELA.metrics import *
    
class DELAModel:
    '''
    A shell of a multi-label classification network with necessary functions
    for multi-label experiments.
    '''
    def __init__(self, configs={}):
        self.configs = configs
        
        # Creating model
        self.net = DELANet(configs)
        self.net.type(self.configs['dtype'])
       
        # Creating learning engine
        self.engine = DELAModelEngine(configs)
    
    def train(self, train_dataloader, val_dataloader=None, quiet_mode=False):
        '''
        Parameters
        ----------
        train_dataloader :
            Loader for Training data.
        val_dataloader :
            Loader for validation data. The default is None.
        quiet_mode : Bool, optional
            If quiet_mode=True, no training information will be displayed. The default is False.
            
        '''
        self.net.training_start(train_dataloader, val_dataloader)
        optimizer, scheduler = self.net.configure_optimizers()
        # Training
        self.engine.learn(self, train_dataloader, val_dataloader, optimizer, scheduler,
                          self.configs['start_epoch'], self.configs['max_epoch'], quiet_mode)
    
    def training_step(self, X, y):
        '''
        Forward model and compute loss for a batch of data during training.
        '''
        output = self.net(X)
        loss_dict = self.net.loss_function_train(output, y)
        
        return loss_dict
    
    def validation_step(self, X, y):
        '''
        Forwarding model and computing loss for a batch of data during validation.
        '''
        output = self.net(X)
        loss_dict = self.net.loss_function_eval(output, y)
        pred_probs = output[-1].sigmoid_()
        pred_labels = (pred_probs > 0.5).type_as(pred_probs)
        loss_dict['AveragePrecision'] = AveragePrecision(pred_probs, y)
        loss_dict['HammingLoss'] = HammingLoss(pred_labels, y)
        
        return loss_dict
    
    def compare_metric(self, m1, m2):
        if m1['AveragePrecision'] > m2['AveragePrecision']:
            return True
        
        if math.fabs(m1['AveragePrecision'] - m2['AveragePrecision']) < self.configs['eps']:
            if m1['HammingLoss'] < m2['HammingLoss']:
                return True
            
            if math.fabs(m1['HammingLoss'] - m2['HammingLoss']) < self.configs['eps']:
                return m1['Loss'] < m2['Loss']
        return False
    
    def test(self, dataloader):
        self.net.eval()
        with torch.no_grad():
            targets = []
            pred_labels = []
            pred_probs = []
            for (X, y) in dataloader:
                X = X.to(self.configs['device'])
                y = y.to(self.configs['device'])
                
                pred_label, pred_prob = self.predict(X)
                targets.append(y)
                pred_labels.append(pred_label)
                pred_probs.append(pred_prob)
            targets = torch.cat(targets, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)
            pred_probs = torch.cat(pred_probs, dim=0)

        return self.evaluate(pred_labels, pred_probs, targets)

    def test_labels(self, dataloader):
        self.net.eval()
        with torch.no_grad():
            targets = []
            pred_labels = []
            pred_probs = []
            for (X, y) in dataloader:
                X = X.to(self.configs['device'])
                y = y.to(self.configs['device'])

                pred_label, pred_prob = self.predict(X)
                targets.append(y)
                pred_labels.append(pred_label)
                pred_probs.append(pred_prob)
            targets = torch.cat(targets, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)
            pred_probs = torch.cat(pred_probs, dim=0)

        return np.array(pred_labels.cpu()), np.array(pred_probs.cpu()), np.array(targets.cpu())
    
    def predict(self, X):
        '''
        Parameters
        ----------
        X : Tensor
            MxN Tensor, the ith instance is stored in X[i,:].

        Returns
        -------
        pred_labels : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            pred_labels[i,j] equals to +1, otherwise pred_labels[i,j] equals to 0.
        pred_probs : Tensor
            MxQ Tensor, the probability of the ith instance belonging to the 
            jth class is stored in pred_probs[i,j]
        '''
        return self.net.predict(X)
    
    def evaluate(self, pred_labels, pred_probs, y):
        '''
        Parameters
        ----------
        pred_labels : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            pred_labels[i,j] equals to +1, otherwise pred_labels[i,j] equals to 0.
        pred_probs : Tensor
            MxQ Tensor, the probability of the ith instance belonging to the 
            jth class is stored in pred_probs[i,j]
        y : Tensor
            MxQ Tensor, if the ith instance belongs to the jth class, then
            y[i,j] equals to +1, otherwise y[i,j] equals to 0.

        Returns
        -------
        metrics : dict
            Metrics for evaluation.
        '''
        metrics = {}
        for metric_name in self.configs['label_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_labels, y)
        for metric_name in self.configs['score_metrics']:
            metrics[metric_name] = eval(metric_name)(pred_probs, y)
        return metrics
    
    def load_checkpoint(self, checkpoint):
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            self.configs['start_epoch'] = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            self.net.to(self.configs['device'])
            self.configs['dtype'] = list(self.net.parameters())[0].dtype
        else:
            print('No checkpoing is found at {}.'.format(checkpoint))
        
    def reset_parameters(self):
        self.net.reset_parameters()