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
from NaiveLinear import NaiveLinear
from NaiveSequential import NaiveSequential

def naive_relu(_in: th.Tensor) -> th.Tensor:
    return th.max(_in, th.zeros_like(_in))

def naive_softmax(_in: th.Tensor) -> th.Tensor:
    return th.exp(_in) / th.sum(th.exp(_in), -1, keepdim=True)

def main() -> None:
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    ic(device)

    model = NaiveSequential(
        NaiveLinear(28 * 28, 512, naive_relu),
        NaiveLinear(512, 256, naive_relu),
        NaiveLinear(256, 128, naive_relu),
        NaiveLinear(128, 10, lambda x: x),
        learning_rate=5e-6,
    )

    data_train = tv.datasets.MNIST(root=Path.cwd(), train=True, download=True, transform=tv.transforms.ToTensor())
    data_test = tv.datasets.MNIST(root=Path.cwd(), train=False, download=True, transform=tv.transforms.ToTensor())

    train = data_train.data.reshape(-1, 28 * 28).to(th.float32) / 255
    train = train.to(device)

    train_labels = data_train.targets
    train_labels = train_labels.to(device)

    test = data_test.data.reshape(-1, 28 * 28).to(th.float32) / 255
    test = test.to(device)

    test_labels = data_test.targets
    test_labels = test_labels.to(device)

    model.testing = test
    model.test_labels = test_labels
    model.to(device)
    model.fit(train, train_labels, epochs=20)

if __name__ == '__main__':
    main()