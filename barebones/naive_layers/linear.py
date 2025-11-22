"""
Defines structure of layers to be used in NaiveSequential
"""

import math
from typing import List, override

import torch as th
from torch import Tensor

from barebones.naive_layers import abstract


class NaiveLinear(abstract.AbstractLayer):
    r"""
    A simple linear layer, the exact calculation is following: $W_h \cdot W_i + b$,
    where $W_i$ is input tensor and
    $W_h$ is hidden state tensor (what the layer has learned) with
    $b$ being bias.

    Args:
        input_size: size of input features
        output_size: size of output features
        batch_normalization: whether to use batch normalization

    Note:
        Batch normalization is applying zscore normalization to linear transformation.
    """

    def __init__(
            self, input_size: int, output_size: int, batch_normalization: bool = False
    ) -> None:
        super().__init__()

        self.weights = th.randn((input_size, output_size), dtype=th.float32, requires_grad=True)
        self.weights.data.mul_(math.sqrt(2. / input_size))

        self.output_size = output_size

        self.bias = th.zeros((output_size,), dtype=th.float32, requires_grad=True)

        # self.activation = activation

        self.batch_normalization = batch_normalization
        self.__training = True

        if self.batch_normalization:
            self.gamma = th.ones(self.output_size, dtype=th.float32, requires_grad=True)
            self.beta = th.zeros(self.output_size, dtype=th.float32, requires_grad=True)

            self.running_mean = th.zeros(self.output_size, dtype=th.float32)
            self.running_var = th.ones(self.output_size, dtype=th.float32)
            self.momentum = 0.1
        else:
            self.gamma = None
            self.beta = None

    def __call__(self, inputs: Tensor) -> Tensor:
        lino = inputs @ self.weights + self.bias

        if self.batch_normalization:
            self.__helper_call_calculations(lino)

        return lino

    def __helper_call_calculations(self, lino: Tensor) -> Tensor:
        assert self.gamma is not None and self.beta is not None

        if self.__training:
            batch_mean = th.mean(lino, dim=0, keepdim=True)
            batch_var = th.var(lino, dim=0, unbiased=False, keepdim=True)

            with ((th.no_grad())):
                self.running_mean = (
                        (1 - self.momentum) * self.running_mean +
                        self.momentum * batch_mean.squeeze()
                )
                self.running_var = (
                        (1 - self.momentum) * self.running_var +
                        self.momentum * batch_var.squeeze()
                )

            norm = (lino - batch_mean) / th.sqrt(batch_var + 1e-6)
        else:
            norm = (lino - self.running_mean) / th.sqrt(self.running_var + 1e-6)

        scaled = self.gamma * norm + self.beta
        return scaled

    def train(self, train: bool):
        self.__training = train

    @property
    @override
    def layers(self) -> None:
        raise NotImplementedError("This layer cannot have sub-layers")

    @property
    def parameters(self) -> List[Tensor]:
        """
        Intended to give parameters of this layer, it is usually called from Sequential

        Returns:
            A list containing weights and bias (and additionally gamma and beta for normalization)

        """
        params = [self.weights, self.bias]
        if self.batch_normalization:
            assert self.gamma is not None and self.beta is not None
            params.extend([self.gamma, self.beta])

        return params

    def to(self, device: th.device):
        """
        Helper method used to send current layer's tensors to device

        Imitates torch tensor's to() method

        Args:
            device: a string name for device to which send the tensors

        Returns:

        """

        # self.weights.data = self.weights.data.to(device)
        # self.bias.data = self.bias.data.to(device)
        self.weights = self.weights.to(device).detach().requires_grad_(True)
        self.bias = self.bias.to(device).detach().requires_grad_(True)

        if self.batch_normalization:
            assert self.gamma is not None and self.beta is not None
            self.gamma = self.gamma.to(device).detach().requires_grad_(True)
            self.beta.data = self.beta.to(device).detach().requires_grad_(True)

            self.running_var = self.running_var.to(device)
            self.running_mean = self.running_mean.to(device)

        return self
