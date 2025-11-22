import numpy as np
from naive_layers import abstract


class ExponentialDecay(interface.BaseLearningRateScheduler[int]):
    def __init__(self, init_lr: float, decay_rate: float):
        """init

        Args:
            init_lr (float): initial learning rate
            decay_rate (float): the rate of decay per epoch
        """
        self.init_lr = init_lr
        self.decay_rate = decay_rate

    def step(self, n: int):
        return self.init_lr * np.exp(-self.decay_rate * n)
