from naive_layers import abstract


class StepDecay(interface.BaseLearningRateScheduler[int]):
    def __init__(self, init_lr: float, decay_rate: float, decay_steps: int):
        self.init_lr = init_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, n: int):
        num_steps = n // self.decay_steps

        return self.init_lr * self.decay_rate ** num_steps
