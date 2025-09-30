import torch
import torchnet
import os

class Engine():
    '''
    Engine to optimize a model
    '''
    
    def __init__(self, configs={}):
        # Training params
        self.device = configs['device']
        
        # Checkpoint saving params
        self.save_checkpoint_path = configs['save_checkpoint_path']
        if not os.path.exists(self.save_checkpoint_path):
            os.makedirs(self.save_checkpoint_path)
        self.dataset_name = configs['dataset_name']
        
        # Other params
        self.display_freq = configs['display_freq']
        
    def learn(self, model, train_dataloader, val_dataloader=None,
              optimizer=None, scheduler=None,
              start_epoch=0, max_epoch=30, quiet_mode=False):
        if optimizer is None:
            optimizer = torch.optim.Adam(model.net.parameters())
        
        # Moving model to right device
        model.net.to(self.device)
        
        # Saving some training information
        self.state = {}
        
        self.quiet_mode = quiet_mode
        if not self.quiet_mode:
            print('Training begins ......')
        
        self.max_epoch = max_epoch
        for epoch in range(start_epoch, max_epoch):
            self.state['epoch'] = epoch
            self.state['lr'] = optimizer.param_groups[0]['lr']
            
            # Training for one epoch
            self._train(model, train_dataloader, optimizer, scheduler)
            
            # Evaluating on validation set
            if val_dataloader is not None:
                is_best, _ = self._validate(model, val_dataloader)
            else:
                is_best = False
            
            # Saving checkpoint
            self._save_checkpoint(model, is_best)
            
            if scheduler is not None:
                scheduler.step_epoch(epoch+1)
        
        if not self.quiet_mode:
            print('Training finishes.')
    
    def _train(self, model, train_dataloader, optimizer, scheduler=None):
        model.net.train()
        
        self._training_epoch_start()
        self.state['max_iters'] = len(train_dataloader)
        for i, (X, y) in enumerate(train_dataloader):
            self.state['iteration'] = i
            X = X.to(self.device)
            y = y.to(self.device)
            
            loss_dict = model.training_step(X, y)
            loss = loss_dict['Loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self._training_save_infors(loss_dict)
            
            # Displaying training information for one iteration
            self._training_step_end()
            
            if scheduler is not None:
                scheduler.step_iter(self.state['epoch']*self.state['max_iters']+i+1)
                self.state['lr'] = optimizer.param_groups[0]['lr']
            
        # Displaying training information for one epoch
        self._training_epoch_end()
        
    def _training_epoch_start(self):
        for key in self.state:
            if isinstance(self.state[key], torchnet.meter.AverageValueMeter):
                self.state[key].reset()
        
    def _training_epoch_end(self):
        '''
        Displaying information after one epoch finishing.
        '''
        if not self.quiet_mode:
            print('Epoch: [{}]\t lr {:.2e}\t Average Loss on training set {:.4f}'.
                  format(self.state['epoch'],
                         self.state['lr'],
                         self.state['Loss'].value()[0]))
    
    def _training_step_end(self):
        '''
        Displaying information after one step.
        '''
        if not self.quiet_mode and self.state['iteration'] % self.display_freq == 0:
            print('Epoch: [{}][{}/{}]\t Loss on training set {:.4f} ({:.4f})'.
                  format(self.state['epoch'],
                         self.state['iteration'],
                         self.state['max_iters'],
                         self.state['Loss'].val,
                         self.state['Loss'].value()[0]))
            
    def _training_save_infors(self, infors):
        for key in infors:
            if key not in self.state:
                self.state[key] = torchnet.meter.AverageValueMeter()
            self.state[key].add(infors[key].detach().item())
        
    def _validate(self, model, val_dataloader):
        model.net.eval()
        with torch.no_grad():
            self._validation_epoch_start()
            self.state['val_max_iters'] = len(val_dataloader)
            for i, (X, y) in enumerate(val_dataloader):
                self.state['iteration'] = i
                X = X.to(self.device)
                y = y.to(self.device)
                
                val_metric = model.validation_step(X, y)
                
                self._validation_save_infors(val_metric)
                
            # Displaying validation information for one epoch
            val_metric = self._validation_epoch_end()
            
            # Comparing with best model
            is_best = self._is_best_model(model, val_metric)
            
        return is_best, val_metric
    
    def _validation_epoch_start(self):
        for key in self.state:
            if isinstance(self.state[key], torchnet.meter.AverageValueMeter):
                self.state[key].reset()
        
    def _validation_epoch_end(self):
        '''
        Displaying information after one epoch finishing.
        '''
        if not self.quiet_mode:
            print('Validation: \t Average Loss {:.4f}'.
                  format(self.state['Loss'].value()[0]))
            
        return {'Loss': self.state['Loss'].value()[0]}
    
    def _validation_step_end(self):
        '''
        Displaying information after one step.
        '''
        if not self.quiet_mode and self.state['iteration'] % self.display_freq == 0:
            print('Validation: [{}/{}]\t Loss {:.4f} ({:.4f})'.
                  format(self.state['iteration'],
                         self.state['val_max_iters'],
                         self.state['Loss'].val,
                         self.state['Loss'].value()[0]))
            
    def _validation_save_infors(self, infors):
        for key in infors:
            if key not in self.state:
                self.state[key] = torchnet.meter.AverageValueMeter()
            self.state[key].add(infors[key])
            
    def _is_best_model(self, model, metric):
        if 'best_val_metric' in self.state and not model.compare_metric(metric,
                                                self.state['best_val_metric']):
            return False
        
        self.state['best_val_metric'] = metric
        return True
    
    def _save_checkpoint(self, model, is_best=False):
        '''
        Saving checkpoint.
        '''
        if is_best:
            filename = self.dataset_name + '_best_checkpoint.pth'
            filename = os.path.join(self.save_checkpoint_path, filename)
            model.configs['best_checkpoint_path'] = filename
            model.configs['best_epoch'] = self.state['epoch']+1
        
            if not self.quiet_mode:
                print('saving model to {}'.format(filename))
            torch.save({'epoch': self.state['epoch']+1,
                        'state_dict': model.net.state_dict()}, filename)
            
class DELAModelEngine(Engine):
    '''
    Engine to optimize DELAModel.
    '''
    def _training_epoch_end(self):
        '''
        Displaying information after one epoch finishing.
        '''
        if not self.quiet_mode:
            print('Epoch: [{}]\t lr {:.2e}\t Average Loss {:.4f}, Cls Loss {:.4f}, '
                  'Kl Loss {:.4f}'.
                  format(self.state['epoch'],
                         self.state['lr'],
                         self.state['Loss'].value()[0],
                         self.state['Cls_loss'].value()[0],
                         self.state['Kl_loss'].value()[0]))
            
    def _validation_epoch_end(self):
        '''
        Displaying information after one epoch finishing.
        '''
        if not self.quiet_mode:
            print('Validation: \t Average Loss {:.4f}\t AP {:.4f}\t HammingLoss {:.4f}'.
                  format(self.state['Loss'].value()[0],
                         self.state['AveragePrecision'].value()[0],
                         self.state['HammingLoss'].value()[0]))

        return {'Loss': self.state['Loss'].value()[0],
                'AveragePrecision': self.state['AveragePrecision'].value()[0],
                'HammingLoss': self.state['HammingLoss'].value()[0]}