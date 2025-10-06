import torch as th
from torch import Tensor
from NaiveLinear import NaiveLinear
from typing import List
from NaiveLRS import CosineAnnealingLR
from BatchGenerator import BatchGenerator

class NaiveSequential:
    def __init__(self, *layers: NaiveLinear, learning_rate: float = 1e-3):
        self.layers = layers
        self.testing: None | Tensor = None
        self.test_labels: None | Tensor = None
        self.learning_rate = learning_rate
        self.scheduler = CosineAnnealingLR(1e-6, 1e-2, pinnacle=20)

        self.loss_collection: List[Tensor] = []
        self.acc_collection: List[float] = []
        self.lr_collection: List[float] = []



    def __call__(self, inputs: Tensor):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def to(self, device: th.device):
        for layer in self.layers:
            layer.to(device=device)

        return self

    @property
    def all_parameters(self) -> List[Tensor]:
        all_params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters)

        return all_params

    def one_training_step(self, images, labels, current_lr: float):
        every = self.all_parameters

        predictions = self(images)

        loss = th.nn.functional.cross_entropy(predictions, labels)

        loss.backward()

        with th.no_grad():
            for idx, parameter in enumerate(every):
                parameter -= parameter.grad * current_lr
                parameter.grad.zero_()


        return loss

    @staticmethod
    def equal(a: Tensor, b: Tensor):
        a_exp = a.unsqueeze(1)

        c = a_exp == b
        c = c.all(-1)
        print(c)

        non_repeat_mask = ~c.any(-1)
        print(a[non_repeat_mask])

    def fit(self, images: Tensor, labels: Tensor, epochs: int, batch_size: int = 128, verbose: bool = False):
        old_loss = 0.
        old_acc = 0.

        for epoch in range(epochs):
            batch_generator = BatchGenerator(images, labels, batch_size)
            current_lr = self.scheduler.step(epoch)
            for idx, values in enumerate(batch_generator):
                _images, _labels = values
                self.one_training_step(_images, _labels, current_lr)

            if self.testing is None or self.test_labels is None:
                print("No testing values given, skipping validation...")
                continue

            with th.no_grad():
                test_predictions = self(self.testing)
                loss = th.nn.functional.cross_entropy(test_predictions, self.test_labels)
                rel_loss = abs(loss - old_loss) / old_loss if old_loss != 0 else 0

                predicted = th.argmax(test_predictions, 1)
                correct = th.eq(predicted, self.test_labels).sum().item()
                accuracy = correct / self.test_labels.size(0)
                rel_acc = abs(accuracy - old_acc) / old_acc if old_loss != 0 else 0

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {loss:.4f} ({rel_loss:.4f}) | Test Accuracy: {accuracy:.4f} ({rel_acc:.4f}) | Current LR: {current_lr:e}")

                old_loss = loss
                old_acc = accuracy

                self.loss_collection.append(loss.cpu())
                self.acc_collection.append(accuracy)
                self.lr_collection.append(current_lr)