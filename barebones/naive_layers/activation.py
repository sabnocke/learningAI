from naive_layers import abstract

from torch import Tensor

class NaiveReLU(abstract.BaseLayer):
    def __call__(self, x: Tensor) -> Tensor:
        return x.clamp(min=0)