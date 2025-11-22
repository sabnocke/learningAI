from typing import Optional, Union
from barebones.naive_lrs import abstract
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    momentum: float
    every_nth: int = 1
    weight_decay: float = 0.
    lrs: Optional[abstract.BaseLearningRateScheduler] = None
