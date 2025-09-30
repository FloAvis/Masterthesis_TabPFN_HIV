import torch

from types import *

class StepLRScheduler:
    '''
    Step decay learning rate schedule.
    '''
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: int,
                 decay_rate: float = 0.1,
                 t_in_epoch: bool = True,
                 iters_per_epoch: int = 0,
                 min_lr: float = 0,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0,
                 initialize: bool = True):
        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        
        # Initialize base learning rates
        if initialize:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.t_in_epoch = t_in_epoch
        self.iters_per_epoch = iters_per_epoch
        if t_in_epoch:
            assert iters_per_epoch > 0
        self.min_lr = min_lr
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(lr - warmup_lr_init) / self.warmup_t for lr in self.base_lrs]
        
        # Initialize learning rate
        self.update_groups(self._get_lr(0))
        
    def state_dict(self) -> dict:
        """
        Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict: dict):
        """
        Loads the schedulers state.

        Parameters
        ----------
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        
    def step_epoch(self, epoch: int):
        '''
        Call at the end of each epoch to calculate next epoch's lr.
        '''
        values = self.get_epoch_lr(epoch)
        if values is not None:
            self.update_groups(values)

    def step_iter(self, num_iters: int):
        '''
        Call at the end of each optimizer step to calculate next step's value.
        '''
        values = self.get_iter_lr(num_iters)
        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value
    
    def get_epoch_lr(self, epoch: int):
        if self.t_in_epoch:
            return self._get_lr(epoch)
        else:
            return None

    def get_iter_lr(self, num_iters: int):
        if not self.t_in_epoch:
            return self._get_lr(num_iters)
        else:
            epoch = num_iters / self.iters_per_epoch
            if self.warmup_t and epoch < self.warmup_t:
                return [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            else:
                return None
        
    def _get_lr(self, t):
        if self.warmup_t and t <= self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        elif self.decay_t == 0:
            lrs = self.base_lrs
        else:
            lrs = [base_lr * (self.decay_rate ** ((t-self.warmup_t) // self.decay_t)) for base_lr in self.base_lrs]

        return lrs
    