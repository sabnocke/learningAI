"""
Index file for naive learning rate schedulers
"""

from barebones.naive_lrs.cosine_annealing import CosineAnnealingLR
from barebones.naive_lrs.exponential_decay import ExponentialDecay
from barebones.naive_lrs.step_decay import StepDecay
from barebones.naive_lrs.plateau_lrs import ReduceLROnPlateau
from barebones.naive_lrs.abstract import BaseLearningRateScheduler

__all__ = [
    "ReduceLROnPlateau", "CosineAnnealingLR", "ExponentialDecay",
    "StepDecay", "BaseLearningRateScheduler"
]

__version__ = "0.1.0"
