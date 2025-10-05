import torch as th
from torch import Tensor
from NaiveLinear import NaiveLinear
from typing import List
from NaiveLRS import LRScheduler, CosineAnnealingLR
from BatchGenerator import BatchGenerator

class NaiveSequential:
    def __init__(self, *layers: NaiveLinear, learning_rate: float = 1e-3):
        self.layers = layers
        self.testing: None | Tensor = None
        self.test_labels: None | Tensor = None
        self.learning_rate = learning_rate
        # self.err = False
        # self.errs = set()
        self.scheduler = CosineAnnealingLR(1e-6, learning_rate, 20)



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

    def fit(self, images: Tensor, labels: Tensor, epochs: int, batch_size: int = 128):
        old_images = images

        for epoch in range(epochs):
            batch_generator = BatchGenerator(images, labels, batch_size)
            current_lr = self.scheduler.step(epoch)
            for idx, values in enumerate(batch_generator):
                _images, _labels = values
                # ic(idx)
                # if idx != 0:
                #     ic(self.equal(old_images, _images))

                self.one_training_step(_images, _labels, current_lr)
                old_images = _images


            if self.testing is None or self.test_labels is None:
                print("No testing values given, skipping validation...")
                continue

            with th.no_grad():
                test_predictions = self(self.testing)
                loss = th.nn.functional.cross_entropy(test_predictions, self.test_labels)
                predicted = th.argmax(test_predictions, 1)
                correct = th.eq(predicted, self.test_labels).sum().item()
                accuracy = correct / self.test_labels.size(0)
                print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f} | Current LR: {current_lr:e}")

        # if self.err:
        #     print("Detected errors")
        #     pprint(self.errs)