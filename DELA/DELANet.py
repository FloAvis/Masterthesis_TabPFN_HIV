import torch
import torch.nn as nn
import torch.nn.functional as F

from DELA.utils import init_random_seed
from DELA.type_def import *
from DELA.layers.MLP import MLP
from DELA.gumbel_softmax_trick import gumbel_sigmoid
from DELA.optims.StepLRScheduler import StepLRScheduler

class DELANet(nn.Module):
    def __init__(self, configs):
        super(DELANet, self).__init__()
        self.configs = configs
        init_random_seed(self.configs['rand_seed'])
        
        # Embedding function
        self.encoder = MLP(configs['in_features'], 256, [256, 512], False,
                           configs['drop_ratio'], "relu")
        self.fc_mu = nn.Linear(256, configs['latent_dim'])
        
        # Standard deviation function to parametrize the noise distribution 
        # (share the first three layers with the embedding function)
        self.fc_logvar = nn.Linear(256, configs['latent_dim'])
        
        # Function to parametrize the binary Concrete gates
        self.logit = nn.Parameter(torch.randn(configs['num_classes'], configs['latent_dim']))
        self.scale_layer = nn.Linear(configs['latent_dim'], configs['latent_dim'])
        
        # Classifiers
        self.decoder = MLP(configs['in_features']+configs['latent_dim'], 512,
                           [256], False, nonlinearity="relu")
        self.classifier = nn.Conv1d(configs['num_classes'], configs['num_classes'], 512,
                                    groups=configs['num_classes'])
        
        # Move model to the right device for consistent initialization
        self.to(configs['device'])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init_random_seed(self.configs['rand_seed'])
        self.encoder.reset_parameters()
        self.fc_mu.reset_parameters()
        self.fc_logvar.reset_parameters()
        self.decoder.reset_parameters()
        self.classifier.reset_parameters()
        self.logit.data.uniform_(-10, 10)
        self.scale_layer.reset_parameters()
        nn.init.constant_(self.scale_layer.bias, 2.0)
    
    def get_config_optim(self):
        return [{'params': self.encoder.parameters()},
                {'params': self.fc_mu.parameters()},
                {'params': self.fc_logvar.parameters()},
                {'params': self.decoder.parameters()},
                {'params': self.classifier.parameters()},
                {'params': self.logit, 'lr': self.configs['lr_ratio']*self.configs['lr']},
                {'params': self.scale_layer.parameters(), 'lr': self.configs['lr_ratio']*self.configs['lr']}]
    
    def forward(self, input: Tensor) -> Tuple[Tensor, ...]:
        # Obtain latent representation of data and standard deviation of the noise distribution [B x D]
        z, n_logvar = self._encode(input)
        
        # Sample the indicator vector of non-informative features for each class label from binary Concrete gates
        if self.training:
            logit = self.scale_layer(self.logit) # [Q x D]
            samples = gumbel_sigmoid(logit, tau=self.configs['tau'], gumbel_noise=True, hard=True) # [Q x D]
             # For numerical stability when calculating the KL-divergence and smoother decision boundary
            samples = samples.clamp(min=self.configs['off_noise']).detach() + samples - samples.detach()
        else:
            samples = None
            
        # Perturb latent representation
        z_k = self._add_noise(z, n_logvar, samples) # [B x Q x D]
        
        # Classification
        preds = self._decode(z_k, input) # [B x Q]
        
        return z, n_logvar, samples, preds
    
    def training_start(self, train_dataloader, val_dataloader=None):
        '''
        Prepare for training.
        '''
        self.iters_per_epoch = len(train_dataloader)
    
    def loss_function_train(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, Kl_loss, Cls_loss = self._compute_loss(*preds, targets) 
        
        return {'Loss': Loss,
                'Kl_loss': Kl_loss,
                'Cls_loss': Cls_loss}
    
    def loss_function_eval(self, preds: Tuple[Tensor, ...], targets: Tensor) -> dict:
        Loss, _, Cls_loss = self._compute_loss(*preds, targets)
        
        return {'Loss': Loss.detach().item(),
                'Cls_loss': Cls_loss.detach().item()}
    
    def predict(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        self.eval()
        with torch.no_grad():
            # Obtain latent representation of data [B x D]
            x_mu, _ = self._encode(input)
            z_x = self._add_noise(x_mu, None, None) # [B x Q x D]
            
            # Classification
            pred_probs = self._decode(z_x, input).sigmoid_() # [B x Q]
            pred_labels = (pred_probs > 0.5).type_as(pred_probs) # [B x Q]
        
        return pred_labels, pred_probs
    
    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = torch.optim.Adam(self.get_config_optim(), lr=self.configs['lr'],
                                     weight_decay=self.configs['weight_decay'])
        if self.configs['lr_scheduler'] == 'step_epoch':
            scheduler = StepLRScheduler(optimizer,
                                        decay_t=self.configs['scheduler_decay_epoch'],
                                        decay_rate=self.configs['scheduler_decay_rate'],
                                        t_in_epoch=True,
                                        iters_per_epoch=self.iters_per_epoch,
                                        warmup_t=self.configs['scheduler_warmup_epoch'])
        else:
            scheduler = None
            
        return optimizer, scheduler
        
    def _encode(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        result = self.encoder(input)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar

    def _add_noise(self, z: Tensor, n_logvar: Tensor, samples: Tensor=None):
        if samples is not None:
            std = torch.exp(0.5 * n_logvar) # sigma = exp(0.5 * log(sigma^2))
            eps = torch.randn_like(std)
            z_k = z.unsqueeze(1) + samples.unsqueeze(0) * std.unsqueeze(1) * eps.unsqueeze(1) # [B x Q x D]
        else:
            z_k = z.unsqueeze(1).expand(-1, self.configs['num_classes'], -1) # [B x Q x D]
            
        return z_k
    
    def _decode(self, z: Tensor, input: Tensor) -> Tensor:
        # Original feature is incorporated for more stable training. Similar technique has been used in Conditional VAE and MPVAE
        z = self.decoder(torch.cat([input.unsqueeze(1).expand(-1, self.configs['num_classes'], -1), z],
                                   dim=2)) # [B x Q x D]
        preds = self.classifier(z).squeeze(2) # [B x Q]
        
        return preds
    
    def _compute_loss(self, z: Tensor, n_logvar: Tensor, samples: Tensor,
                      preds: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        # Classification loss
        Cls_loss = F.multilabel_soft_margin_loss(preds, targets) * targets.size(1)
        
        # KL-divergence loss (Constraint on noise distribution)
        if samples is not None:
            Kl_loss = self._KL(z, n_logvar, samples)
            Loss = Cls_loss + self.configs['beta'] * Kl_loss
        else:
            Kl_loss = None
            Loss = Cls_loss

        return Loss, Kl_loss, Cls_loss
    
    def _KL(self, z: Tensor, n_logvar: Tensor, samples: Tensor):
        z = z.unsqueeze(1)
        n_logvar = n_logvar.unsqueeze(1)
        samples = samples.unsqueeze(0)
        KL_mat = -n_logvar - 2*torch.log(samples+1e-6) - 1 + torch.exp(n_logvar)*samples**2 + z**2 # [B x Q x D]
        
        return torch.mean(0.5*torch.sum(KL_mat, dim=2))