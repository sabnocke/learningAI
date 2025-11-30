from os import read
from typing import Callable, Union, override
import numpy as np

from barebones.naive_lrs import abstract

type Number = Union[int, float]

class CosineAnnealingLR(abstract.BaseLearningRateScheduler):
    """
    An algorithm for scheduling learning rate using cosine function.
    It periodically reaches the **pinnacle** parameter,
    allowing to escape local minima and exploring entire range of lr values.

    Args:
        lr_min (float): The smallest learning rate (final)
        lr_max (float): The largest learning rate (start)
        pinnacle (int): At which epoch should be minimum reached
        cyclic (bool): Sets cyclic learning rate; see Note

    Note:
        If cyclic = False and pinnacle != # epochs, then two scenarios can happen:

            - if pinnacle < # epochs =>
              after minimum is reached the *learning rate* starts to raise again,
              potentially ending with worse results

            - if pinnacle > # epochs => minimum will be never be reached,
              potentially not reaching ideal solution

        Otherwise (cyclic = True)
            -   creates sawtooth graph, repeatedly reaching both maximum and minimum
    """

    def __init__(self, lr_min: float, lr_max: float, pinnacle: int, cyclic: bool = False) -> None:

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.pinnacle = pinnacle
        self.cyclic = cyclic
        self.arg: Callable[[Number], float] = (
            lambda x: np.pi * ((x % pinnacle) if cyclic else x / pinnacle)
        )
        self.lr: Callable[[Number], float] = (
            lambda x: lr_min + 1/2 * (lr_max - lr_min) * (1 + np.cos(self.arg(x)))
        )

    @override
    def _step_int(self, n: int) -> float:
        if self.cyclic:
            relative_epoch = n % self.pinnacle
            t_max = self.pinnacle
        else:
            relative_epoch = n
            t_max = self.pinnacle

        angle = (relative_epoch / t_max) * np.pi
        return self.lr_min + 1/2 * (self.lr_max - self.lr_min) * (1 + np.cos(angle))

    def __str__(self):
        return (f"{self.__class__.__name__}\n"
                f" - lr_min: {self.lr_min}\n"
                f" - lr_max: {self.lr_max}\n"
                f" - pinnacle: {self.pinnacle}\n"
                f" - cyclic: {self.cyclic}"
                )
