import math
import torch as th
import numpy as np

#TODO add more schedulers

class ReduceLROnPlateau:
    def __init__(self, init_lr, factor=0.1, patience=5, mode: str = 'min', verbose: bool = False):
        self.init_lr = init_lr
        self.factor = factor
        self.patience = patience
        self.best_model = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode
        self.num_bad_epochs = 0
        self.current_lr = init_lr
        self.improve = lambda metric: metric < self.best_model if self.mode == 'min' else metric > self.best_model
        self.verbose = verbose

    def step(self, metric):
        is_improving = self.improve(metric)

        if is_improving:
            self.best_model = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.current_lr *= self.factor
            self.num_bad_epochs = 0
            if self.verbose:
                print(f"Reducing learning rate to {self.current_lr:.2e}")

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