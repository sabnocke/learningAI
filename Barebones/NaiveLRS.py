import numpy as np

class ReduceLROnPlateau:
    def __init__(self, init_lr, factor=0.1, patience=5, min_delta: float = 1e-4, mode: str = 'min', verbose: bool = False):
        self.init_lr = init_lr
        self.factor = factor
        self.patience = patience
        self.best_model = float('inf') if mode == 'min' else float('-inf')
        self.mode = mode
        self.num_bad_epochs = 0
        self.current_lr = init_lr
        self.improve = lambda metric: metric < self.best_model if self.mode == 'min' else metric > self.best_model
        self.verbose = verbose
        self.min_delta = min_delta
        self.old_change: None | float = None

    def step(self, metric: float):
        if metric < self.best_model - self.min_delta:
            self.best_model = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.current_lr *= self.factor

        return self.current_lr

    def rel_change(self, metric: float):
        if self.old_change is None:
            self.old_change = metric
            return False

        change = abs(metric - self.old_change) / self.old_change
        self.old_change = metric
        return change < self.min_delta


class StepDecay:
    def __init__(self, init_lr: float, decay_rate: float, decay_steps: int):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, epochs: int):
        num_steps = epochs // self.decay_steps

        return self.init_lr * self.decay_rate ** num_steps

class ExponentialDecay:
    def __init__(self, init_lr: float, decay_rate: float):
        self.init_lr = init_lr
        self.decay_rate = decay_rate

    def step(self, epoch: int):
        return self.init_lr * np.exp(-self.decay_rate * epoch)

class CosineAnnealingLR:
    def __init__(self, lr_min: float, lr_max: float, pinnacle: int, cyclic: bool = False):
        """

        :param lr_min: minimum learning rate, the smallest learning rate ever will be
        :param lr_max: initial or largest learning rate, the largest learning rate will be
        :param pinnacle: at which epoch is will the decay be smallest (i.e. equal to lr_min)
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.pinnacle = pinnacle

        if cyclic:
            self.lr = lambda curr_epoch: lr_min + 1/2 * (lr_max - lr_min) * (1 + np.cos((curr_epoch % pinnacle)/pinnacle * np.pi))
        else:
            self.lr = lambda curr_epoch: lr_min + 1/2 * (lr_max - lr_min) * (1 + np.cos(curr_epoch/pinnacle * np.pi))

    def step(self, epoch: int):
        return self.lr(epoch)
