from typing import Tuple, Union, List

import torch
import torch.nn.functional as F
import torch as th
import numpy as np
from torch import Tensor

from barebones.naive_layers import abstract

type DualInt = int | Tuple[int, int]

def _pair(x: DualInt) -> Tuple[int, int]:
    if isinstance(x, int):
        return x, x
    return x

class NaiveConv2d(abstract.AbstractLayer):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: DualInt,
                 stride: DualInt = 1,
                 padding: DualInt = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size: Tuple[int, int] = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self._kw, self._kh = self.kernel_size
        self._sw, self._sh = self.stride
        self._pw, self._ph = self.padding

        weight_shape = (out_channels, in_channels, self._kh, self._kw)
        self.weights = th.randn(weight_shape, requires_grad=True)

        self.bias = th.zeros(out_channels, requires_grad=True)

        n_in = in_channels * self._kw * self._kh
        self.weights.data.mul_(np.sqrt(2. / n_in))

        self.__training = True

    def to(self, device: th.device) -> "NaiveConv2d":
        self.weights = self.weights.to(device).detach().requires_grad_(True)
        self.bias = self.bias.to(device).detach().requires_grad_(True)

        return self

    def train(self, train: bool = True):
        self.__training = train
        # return self

    @property
    def parameters(self) -> List[Tensor]:
        params = [self.weights, self.bias]
        return params

    def __call__(self, _in: Tensor) -> Tensor:
        assert _in.dim() == 4, f"Expected four dimensions, but got {_in.dim()} {_in.shape}"

        batch_size, _, h, w = _in.shape

        # 1. Unfold the image into patches
        # Input: (B, C, H, W) -> Output: (B, C*K*K, Number_of_Patches)
        inp_unf = F.unfold(_in, self.kernel_size, stride=self.stride, padding=self.padding)

        # 2. Reshape Kernel for Matrix Multiplication
        # We flatten the kernel dimensions: (Out, In, K, K) -> (Out, In*K*K)
        w_flat = self.weights.view(self.out_channels, -1)

        # 3. Perform Convolution as Matrix Multiplication
        # (Out, In*K*K) @ (B, In*K*K, Patches)
        # We use Einstein Summation or explicit matmul.
        # Easier way: transpose inputs to make them line up

        # Result shape: (Batch, Out_Channels, Patches)
        out_unf = inp_unf.transpose(1, 2).matmul(w_flat.t()).transpose(1, 2)

        # 4. Add Bias (Broadcast over patches)
        # bias is (Out_Channels), we need to add it to dim 1
        out_unf += self.bias.view(1, -1, 1)

        # 5. Fold (Reshape) back to image dimensions
        # We need to calculate the new height and width mathematically
        h_out = (h + 2 * self._ph - self._kh) // self._sh + 1
        w_out = (w + 2 * self._pw - self._kw) // self._sw + 1

        del inp_unf
        return out_unf.view(batch_size, self.out_channels, h_out, w_out)
