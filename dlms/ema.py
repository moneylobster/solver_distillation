"""Exponential moving averages of the designer-network parameters.

The paper (Sec. 3.3) tracks EMAs with half-lives of 1, 2 and 3 kimg (plus the
raw weights) and selects the best-performing one by FID after training.
"""

import copy

import torch


class EMA:
    def __init__(self, module, halflife_kimg):
        self.halflife_kimg = halflife_kimg
        self.shadow = {k: v.detach().clone().float()
                       for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module, batch_size):
        beta = 0.5 ** (batch_size / (self.halflife_kimg * 1000.0))
        for k, v in module.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(beta).add_(v.float(), alpha=1 - beta)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow
