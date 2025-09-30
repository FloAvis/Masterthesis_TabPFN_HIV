import torch

def gumbel_sigmoid(logits, tau=2/3, gumbel_noise=True, hard=False):
    '''
    Sample from the binary Concrete distribution and optionally discretize.
    Refer to [1][2] for details.
    
    [1] Jang, E., et al. (2017). Categorical reparameterization with gumbel-softmax.
    Proceedings of the 5th International Conference on Learning Representations. Toulon, France.
    [2] Maddison, C. J., et al. (2017). The concrete distribution: A continuous
    relaxation of discrete random variables. Proceedings of the 5th International
    Conference on Learning Representations. Toulon, France.
    '''
    if gumbel_noise:
        uniforms = clamp_probs(torch.rand(logits.size(), device=logits.device, dtype=logits.dtype))
        samples = uniforms.log() - (-uniforms).log1p() + logits
    else:
        samples = logits
    y_soft = torch.sigmoid(samples/tau)
    
    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard + y_soft - y_soft.detach() # the gradients will only be backpropagated to y_soft
    else:
        # Reparameterization trick.
        ret = y_soft
    return ret

def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)