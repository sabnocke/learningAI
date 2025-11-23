r"""
Defines all available activation functions

The purpose of activation functions is to break linearity (in case of linear layer)
as per mathematics definition any number of linear operation can (and will) "collapse"
into one singular operation ($3 \cdot x$ and then $2 \cdot x$ is equivalent to $6x$),
for the layers it means that any number of linear layers
without activation will collapse into one layer
(mathematically, computationally there is still $n$ layers).

The second purpose has to do with operations available to any purely linear transformation,
those are *scale*, *rotate* and *shear*, due to that parallel lines remain parallel.
Activation functions allow *folding*, *twisting* and *bending* the data space,
thus allowing to create complex decision boundaries (curves, circles, squiggles, ...)
instead of just lines.
"""
from torch import Tensor
from barebones.naive_layers import BaseLayer

class NaiveReLU(BaseLayer):
    r"""
    ReLU (Rectified Linear Unit) $f(x) = \max(0, x)$, where x is input tensor.

    It outputs strict zeros, i.e. the neuron is either active or not
    (for picture of cat, the neurons for car wheels should be silent)

    """
    def __call__(self, x: Tensor) -> Tensor:
        r"""

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: the same tensor but clamped to non-negative values
        """
        return x.clamp(min=0)
