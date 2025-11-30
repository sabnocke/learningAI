
from torch import Tensor
from barebones.naive_layers.abstract import BaseLayer


class NaiveFlatten(BaseLayer):
    def __call__(self, source: Tensor) -> Tensor:
        return source.reshape(source.shape[0], -1)