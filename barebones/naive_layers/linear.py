"""
Defines structure of linear layer to be used in NaiveSequential
"""

import math
from typing import List, override, Never

import torch as th
from torch import Tensor

from barebones.naive_layers import AbstractLayer


class NaiveLinear(AbstractLayer):
    r"""
    Fully connected (dense) layer. Performs $y = xA^T + b$

    Includes integrated batch normalization to stabilize training.

    Args:
        input_size:
            size of input features
            (output size of previous layer or size of input data)
        output_size: size of output features

    Attributes:
        weights (Tensor): Learnable weights of shape (input_size, output_size).
        bias (Tensor): Learnable bias of shape (input_size,).

    Note:
        Batch normalization is applying zscore normalization after linear transformation.

        The He Initialization (or Kaiming Initialization) is a strategy for setting
        the random initial value of model's weights.
        Since during back-propagation value of each layer's weights is
        updated based on previous layer's weights (from last to first).

        If the values are large and there is many of them,
        by adding them it creates even larger values, scaling towards infinity
        - this is called exploding gradients

        If the values are small, it creates increasingly smaller values, scaling towards zero
        - this is called vanishing gradients

        Inevitably either

        - dying
            - the learning is so small the model freezes
            - similar to being stuck on a boat in an ocean without any wind

        - crashing
            - the new values become increasingly larger eventually overflowing their type
            - thus creating a "NaN" virus;
                - any operation with NaN creates another NaN, thus the model becomes unusable

        The various initialization methods (of which He is one)
        are meant to prevent this from happening by "stabilizing"
        the weights during initialization
    """

    def __init__(
            self, input_size: int, output_size: int
    ) -> None:
        super().__init__()

        # He initialization (Kaiming normal)
        self.weights = th.randn((input_size, output_size), dtype=th.float32, requires_grad=True)
        self.weights.data.mul_(math.sqrt(2. / input_size))

        self.output_size = output_size

        self.bias = th.zeros((output_size,), dtype=th.float32, requires_grad=True)

        self.__training = True

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weights + self.bias

    def train(self, train: bool):
        self.__training = train

    @property
    @override
    def layers(self) -> List[Never]:
        return []

    @property
    def parameters(self) -> List[Tensor]:
        """
        Intended to give parameters of this layer, it is usually called from Sequential

        Returns:
            A list containing weights and bias (and additionally gamma and beta for normalization)

        """

        return [self.weights, self.bias]

    def to(self, device: th.device) -> "NaiveLinear":
        """
        Helper method used to send current layer's tensors to device

        Imitates torch tensor's to() method

        Args:
            device: a string name for device to which send the tensors

        Returns:
            Layer: self
        """

        self.weights = self.weights.to(device).detach().requires_grad_(True)
        self.bias = self.bias.to(device).detach().requires_grad_(True)

        return self
