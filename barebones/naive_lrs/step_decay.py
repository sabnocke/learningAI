from typing import override
from naive_lrs import abstract


class StepDecay(abstract.BaseLearningRateScheduler):
    def __init__(self, init_lr: float, decay_rate: float, decay_steps: int):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    @override
    def _step_int(self, n: int) -> float:
        num_steps = n // self.decay_steps
        return self.init_lr * self.decay_rate ** num_steps
