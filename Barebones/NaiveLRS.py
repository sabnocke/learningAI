from typing import Callable
import numpy as np

type Number = int | float

class ReduceLROnPlateau:
    def __init__(self, init_lr, factor=0.1, patience=5, min_delta: float = 1e-4):
        self.init_lr = init_lr
        self.factor = factor
        self.patience = patience
        self.best_model = float('-inf')
        self.num_bad_epochs = 0
        self.current_lr = init_lr
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

# pylint: disable
class CosineAnnealingLR:
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
                If cyclic = False and pinnacle != # of epochs, then two scenarios can happen:
                    if pinnacle < # of epochs := creates a V point in graph,
                    where minimum is reached after which the *lr* starts to raise again,
                    potentially ending with worse results

                    if pinnacle > # of epochs := minimum will be never be reached,
                    potentially not reaching ideal solution
            """

    def __init__(self, lr_min: float, lr_max: float, pinnacle: int, cyclic: bool = False) -> None:

        self.lr_min = lr_min
        self.lr_max = lr_max
        self.pinnacle = pinnacle
        self.arg: Callable[[Number], float] = (
            lambda x: np.pi * ((x % pinnacle) if cyclic else x / pinnacle)
        )
        self.lr: Callable[[Number], float] = (
            lambda x: lr_min + 1/2 * (lr_max - lr_min) * (1 + np.cos(self.arg(x)))
        )

    def step(self, epoch: int) -> float:
        return self.lr(epoch)
