from typing import Tuple, Union
from torch import Tensor
import torch.nn.functional as F
from barebones.naive_layers import BaseLayer

def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return x, x
    return x

class NaiveMaxPool2d(BaseLayer):
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]] = 2,
                 stride: Union[int, Tuple[int, int]] = 2):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self._kw, self._kh = _pair(kernel_size)
        self._sw, self._sh = _pair(stride)

    def __call__(self, source: Tensor) -> Tensor:
        # source: (B, C, H, W)
        b, c, h, w = source.shape

        # Unfold to get windows
        source_reshaped = source.reshape(b * c, 1, h, w)
        windows = F.unfold(source_reshaped, kernel_size=self.kernel_size, stride=self.stride)

        # windows: (B*C, K*K, num_windows)
        # Take max over the window dimension (dim 1)
        max_vals, _ = windows.max(dim=1)

        h_out = (h - self._kh) // self._sh + 1
        w_out = (w - self._kw) // self._sw + 1

        return max_vals.reshape(b, c, h_out, w_out)
