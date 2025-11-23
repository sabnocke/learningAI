import torch as th
from torch import Tensor
from barebones.naive_layers import abstract

class NaiveDropout(abstract.BaseLayer):
    r"""
    Inverted Dropout Layer for regularization.

    Randomly zeroes out elements of the input tensor with probability p
    during training to prevent overfitting.

    ##Math
    - Training: $\text{output} = \frac{\text{input} \cdot \text{mask}}{1 - p}

    - Inference: output = input
    """

    def __init__(self, p: float = 0.5):
        """

        Args:
            p: Probability to zero out elements of the input tensor, default: 0.5
        """
        super().__init__()
        self.p = p

        assert 0 <= self.p <= 1, "Dropout probability must be in [0,1["

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        mask = th.bernoulli(th.full_like(x, 1 - self.p))

        output = x * mask

        return output / (1 - self.p)
