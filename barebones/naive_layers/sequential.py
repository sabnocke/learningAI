from typing import List, Optional, Tuple, override
import torch as th
from torch import Tensor
from barebones.naive_lrs import CosineAnnealingLR
from barebones.aux import TrainingConfig
from naive_layers import abstract

type Layer = abstract.AbstractLayer

class NaiveSequential(abstract.AbstractLayer):
    def __init__(self, *layers: Layer, config: TrainingConfig):
        super().__init__()

        self._layers = layers
        self.testing: Optional[Tensor] = None
        self.test_labels: Optional[Tensor] = None
        self.learning_rate = config.learning_rate
        self.config = config
        self.scheduler = CosineAnnealingLR(1e-6, 1e-2, pinnacle=20)

        self.loss_collection: List[Tensor] = []
        self.acc_collection: List[float] = []
        self.lr_collection: List[float] = []
        self.verbose: bool = False

    @property
    @override
    def layers(self) -> Tuple[Layer, ...]:
        return self._layers

    def __call__(self, inputs: Tensor):
        for layer in self._layers:
            inputs = layer(inputs)

        return inputs

    def to(self, device: th.device):
        """
        Helper method to send all layers to given device

        Args:
            device (th.device):
                device to send all layers to

        Returns: self
        """
        for layer in self._layers:
            layer.to(device=device)

        return self

    @property
    def parameters(self) -> List[Tensor]:
        all_params = []
        for layer in self._layers:
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters)

        return all_params

    def train(self, train: bool):
        for layer in self._layers:
            layer.train(train)
