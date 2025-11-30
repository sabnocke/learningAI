from typing import List

import torch as th
from torch import Tensor

from barebones.naive_layers import BaseLayer


class NaiveBatchNorm1d(BaseLayer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = True

        self.gamma = th.ones(num_features, requires_grad=True)
        self.beta = th.zeros(num_features, requires_grad=True)

        self.running_mean = th.zeros(num_features)
        self.running_var = th.ones(num_features)

    def train(self, train=True):
        self.training = train

    def __call__(self, source: th.Tensor):
        if self.training:
            batch_mean = th.mean(source, dim=0, keepdim=True)
            batch_var = th.var(source, dim=0, keepdim=True, unbiased=False)

            with th.no_grad():
                self.running_mean = (
                        (1 - self.momentum) * self.running_mean +
                        self.momentum * batch_mean.squeeze()
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var +
                    self.momentum * batch_var.squeeze()
                )

            norm = (source - batch_mean) / th.sqrt(batch_var + self.eps)
        else:
            norm = (source - self.running_mean) / th.sqrt(self.running_var + self.eps)

        return self.gamma * norm + self.beta

    @property
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


    def to(self, device: th.device):
        self.gamma = self.gamma.to(device).detach().requires_grad_(True)
        self.beta = self.gamma.to(device).detach().requires_grad_(True)

        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)

        return self

class NaiveBatchNorm2d(BaseLayer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable Parameters (Gamma and Beta)
        # Shape: (1, C, 1, 1) to allow broadcasting over Batch, Height, and Width
        self.gamma = th.ones((1, num_features, 1, 1), requires_grad=True)
        self.beta = th.zeros((1, num_features, 1, 1), requires_grad=True)

        # Running Stats (Non-learnable buffers)
        # We track these separately for Inference
        self.running_mean = th.zeros(num_features)
        self.running_var = th.ones(num_features)

    def __call__(self, inputs: Tensor) -> Tensor:
        if self.training:
            batch_mean = th.mean(inputs, dim=(0, 2, 3))
            batch_var = th.var(inputs, dim=(0, 2, 3), unbiased=False)

            # 2. Update Running Stats (Momentum)
            # We detach to ensure this doesn't become part of the computation graph
            with th.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * batch_var

            # 3. Reshape for Broadcasting
            # (C) -> (1, C, 1, 1) so it subtracts correctly from (N, C, H, W)
            mean = batch_mean.view(1, self.num_features, 1, 1)
            var = batch_var.view(1, self.num_features, 1, 1)
        else:
            mean = self.running_mean.view(1, self.num_features, 1, 1)
            var = self.running_var.view(1, self.num_features, 1, 1)

        norm = (inputs - mean) / th.sqrt(var + self.eps)
        return self.gamma * norm + self.beta

    def to(self, device: th.device):
        # Move params (with gradients)
        self.gamma = self.gamma.to(device).detach().requires_grad_(True)
        self.beta = self.beta.to(device).detach().requires_grad_(True)

        # Move buffers (no gradients)
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self

    @property
    def parameters(self):
        return [self.gamma, self.beta]

    def train(self, train=True):
        self.training = train
