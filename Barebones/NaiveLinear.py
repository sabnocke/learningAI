import torch as th
import math
from typing import Callable

class NaiveLinear:
    def __init__(self, input_size: int, output_size: int, activation: Callable, batch_normalization: bool = False):
        self.weights = th.randn((input_size, output_size), dtype=th.float32, requires_grad=True)
        self.weights.data.mul_(math.sqrt(2. / input_size))

        self.output_size = output_size

        self.bias = th.zeros((output_size,), dtype=th.float32, requires_grad=True)

        self.activation = activation

        self.has_weights = True
        self.has_bias = True
        self.has_gamma = True
        self.has_beta = True
        self.batch_normalization = batch_normalization

        if self.batch_normalization:
            self.gamma = th.ones(self.output_size, dtype=th.float32, requires_grad=True)
            self.beta = th.zeros(self.output_size, dtype=th.float32, requires_grad=True)
        else:
            self.gamma = None
            self.beta = None

    def __call__(self, inputs: th.Tensor):
        lino = inputs @ self.weights + self.bias

        if self.batch_normalization:
            batch_mean = th.mean(lino, dim=0, keepdim=True)
            batch_var = th.var(lino, dim=0, unbiased=False, keepdim=True)

            norm = (lino - batch_mean) / th.sqrt(batch_var + 1e-6)
            scaled = self.gamma * norm + self.beta

            return self.activation(scaled)

        return self.activation(lino)

    def reform(self):
        self.gamma = th.ones(self.output_size, dtype=th.float32, requires_grad=True)
        self.beta = th.zeros(self.output_size, dtype=th.float32, requires_grad=True)
        self.batch_normalization = True

    @property
    def parameters(self):
        params = [self.weights, self.bias]
        if self.batch_normalization:
            params.extend([self.gamma, self.beta])

        return params

    def to(self, device: th.device):
        self.weights.data = self.weights.data.to(device)
        self.bias.data = self.bias.data.to(device)
        if self.batch_normalization:
            self.gamma.data = self.gamma.data.to(device)
            self.beta.data = self.beta.data.to(device)

        return self
