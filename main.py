from pprint import pprint

from torch import Tensor
import torch as th
import torchvision as tv
import math
from typing import Callable, List
from pathlib import Path
from icecream import ic
import numpy as np
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size: int = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images: Tensor = images
        self.labels: Tensor = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.images) / self.batch_size)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.images):
            raise StopIteration

        images = self.images[self.index: self.index + self.batch_size].clone()
        labels = self.labels[self.index: self.index + self.batch_size].clone()
        self.index += self.batch_size
        return images, labels

class NaiveLinear:
    def __init__(self, input_size: int, output_size: int, activation: Callable, batch_normalization: bool = False):
        self.w_init = th.randn((input_size, output_size), dtype=th.float32, requires_grad=True)
        self.w_init.data.mul(math.sqrt(2. / input_size))

        b_shape = (output_size,)
        self.b_init = th.zeros(b_shape, dtype=th.float32, requires_grad=True)

        self.activation = activation

        self.has_weights = True
        self.has_bias = True
        self.has_gamma = True
        self.has_beta = True
        self.batch_normalization = batch_normalization

        if batch_normalization:
            self.gamma = th.ones(output_size, dtype=th.float32, requires_grad=True)
            self.beta = th.zeros(output_size, dtype=th.float32, requires_grad=True)
        else:
            self.gamma = None
            self.beta = None

    def __call__(self, inputs: th.Tensor):
        lino = inputs @ self.w_init + self.b_init

        if self.batch_normalization:
            batch_mean = th.mean(lino, dim=0, keepdim=True)
            batch_var = th.var(lino, dim=0, unbiased=False, keepdim=True)

            norm = (lino - batch_mean) / th.sqrt(batch_var + 1e-6)
            scaled = self.gamma * norm + self.beta

            return self.activation(scaled)

        return self.activation(lino)

    @property
    def parameters(self):
        params = [self.w_init, self.b_init]
        if self.batch_normalization:
            params.extend([self.gamma, self.beta])

        return params

class NaiveSequential:
    def __init__(self, *layers: NaiveLinear, learning_rate: float = 1e-3):
        self.layers = layers
        self.testing: None | th.Tensor = None
        self.test_labels: None | th.Tensor = None
        self.learning_rate = learning_rate
        self.err = False
        self.errs = set()

    def __call__(self, inputs: th.Tensor):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    @property
    def all_parameters(self) -> List[th.Tensor]:
        all_params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters)

        return all_params

    def one_training_step(self, images, labels):
        every = self.all_parameters

        predictions = self(images)

        loss = th.nn.functional.cross_entropy(predictions, labels)

        loss.backward()

        with th.no_grad():
            for idx, parameter in enumerate(every):
                parameter -= parameter.grad * self.learning_rate
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
            for idx, values in enumerate(batch_generator):
                _images, _labels = values
                # ic(idx)
                # if idx != 0:
                #     ic(self.equal(old_images, _images))

                self.one_training_step(_images, _labels)
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
                print(f"Epoch {epoch + 1}/{epochs} | Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

        if self.err:
            print("Detected errors")
            pprint(self.errs)


def naive_relu(_in: th.Tensor) -> th.Tensor:
    return th.max(_in, th.zeros_like(_in))

def naive_softmax(_in: th.Tensor) -> th.Tensor:
    return th.exp(_in) / th.sum(th.exp(_in), -1, keepdim=True)

def main() -> None:

    # device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    model = NaiveSequential(
        NaiveLinear(28 * 28, 512, naive_relu),
        NaiveLinear(512, 256, naive_relu),
        NaiveLinear(256, 128, naive_relu),
        # NaiveLinear(128, 10, naive_relu),
        NaiveLinear(128, 10, lambda x: x),
        learning_rate=1e-5,
    )

    data_train = tv.datasets.MNIST(root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor())
    data_test = tv.datasets.MNIST(root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor())

    train = data_train.data.reshape(-1, 28 * 28).to(th.float32) / 255

    train_labels = data_train.targets

    test = data_test.data.reshape(-1, 28 * 28).to(th.float32) / 255

    test_labels = data_test.targets

    model.testing = test
    model.test_labels = test_labels
    model.fit(train, train_labels, epochs=20)

if __name__ == '__main__':
    main()