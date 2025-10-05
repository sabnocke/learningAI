import math
import torch as th
import numpy as np

#TODO add more schedulers

class LRScheduler:
    def __init__(self, init_lr: float, decay_rate: float, decay_steps: int):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, epochs: int):
        num_steps = epochs // self.decay_steps

        return self.init_lr * self.decay_rate ** num_steps


class CosineAnnealingLR:
    def __init__(self, lr_min: float, lr_max: float, pinnacle: int):
        """

        :param lr_min: minimum learning rate, the smallest learning rate ever will be
        :param lr_max: initial or largest learning rate, the largest learning rate will be
        :param pinnacle: at which epoch is will the decay be smallest (i.e. equal to lr_min)
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.pinnacle = pinnacle

        self.lr = lambda curr_epoch: lr_min + 1/2 * (lr_max - lr_min) * (1 + curr_epoch/pinnacle * np.pi)

    def step(self, epoch: int):
        return self.lr(epoch)