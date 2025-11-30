from typing import override, TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from barebones.naive_lrs.abstract import BaseLearningRateScheduler

class ExponentialDecay(BaseLearningRateScheduler):
    def __init__(self, init_lr: float, decay_rate: float):
        """init

        Args:
            init_lr (float): initial learning rate
            decay_rate (float): the rate of decay per epoch
        """
        self.init_lr = init_lr
        self.decay_rate = decay_rate

    @override
    def _step_int(self, n: int) -> float:
        return self.init_lr * np.exp(-self.decay_rate * n)
