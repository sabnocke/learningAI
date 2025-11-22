from typing import Any, List, Optional, Union
from functools import singledispatchmethod
from torch import Tensor
import torch as th
from barebones.naive_lrs import abstract
from barebones.aux import TrainingConfig

class NaiveOptimizer:
    def __init__(
            self, parameters: List[Tensor], config: TrainingConfig ):
        self.parameters = parameters
        self.config = config
        self.lr = config.learning_rate

        self.velocities = [th.zeros_like(p) for p in self.parameters]

    @singledispatchmethod
    def step(self, n) -> Any:
        raise NotImplementedError(f"Optimizer does not support step with argument type: {type(n)}")

    @step.register
    def _(self, n: None = None) -> None:
        del n   # n isn't used
        self._apply_step(self.lr)

    @step.register
    def _(self, n: Union[int, float]):
        assert self.config.lrs is not None, "Scheduler must be initialized before calling step"
        self._apply_step(self.config.lrs.step(n))

    def zero_grad(self) -> None:
        with th.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.zero_()

    def _apply_step(self, current_lr: float):
        with th.no_grad():
            for i, param in enumerate(self.parameters):
                if param.grad is None:
                    continue

                grad = param.grad

                # 1. Apply Weight Decay (L2 Regularization)
                if self.config.weight_decay > 0.0:
                    grad = grad + self.config.weight_decay * param

                # 2. Apply Momentum
                if self.config.momentum > 0.0:
                    # v = momentum * v - lr * grad
                    # Note: There are slightly different formulas.
                    # PyTorch standard is: v = momentum * v + grad; w = w - lr * v

                    self.velocities[i] = self.config.momentum * self.velocities[i] + grad
                    update = self.velocities[i]

                    param -= current_lr * update
                else:
                    param -= current_lr * grad
