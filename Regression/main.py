import pandas as pd
from dataclasses import dataclass
from functools import wraps
from time import time
from typing import Callable
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from Model import ANN

df = pd.read_csv("../datasets/BostonHousing.csv")

type Maybe[T] = T | None


@dataclass
class TrainingData:
    train_x: torch.Tensor
    train_y: Maybe[torch.Tensor]
    test_x: torch.Tensor
    test_y: Maybe[torch.Tensor]


@dataclass
class TrainingModel[TOptim, TLoss]:
    model: nn.Module
    optimizer: TOptim
    loss: TLoss


def measure(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Elapsed time: {time() - start:.4f}s")
        return result

    return wrapper


def prepare_model(in_dims: int):
    model = ANN(in_dims)
    loss = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    return TrainingModel(
        model=model,
        loss=loss,
        optimizer=optimizer
    )


def prepare_data():
    x = df.drop(["medv"], axis=1).values
    y = df["medv"].values

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return TrainingData(
        train_x=torch.from_numpy(train_x.astype(np.float32)),
        test_x=torch.from_numpy(test_x.astype(np.float32)),
        train_y=torch.from_numpy(train_y.astype(np.float32)).view(-1,1),
        test_y=torch.from_numpy(test_y.astype(np.float32))
    )


@measure
def no_fold_training_loop(n_epochs: int = 100, in_dims: int = 64, print_freq: int = 10, patience: float = 3e-3):
    data = prepare_data()

    model: TrainingModel[torch.optim.RMSprop, torch.MSELoss] = prepare_model(13)

    old_loss: float = float("inf")

    try:
        for i in range(n_epochs):
            model.optimizer.zero_grad()
            out = model.model(data.train_x)
            loss = model.loss(out, data.train_y)
            loss.backward()
            model.optimizer.step()

            rel_loss = abs(loss - old_loss) / old_loss if old_loss != 0 else 0

            if (i + 1) % print_freq == 0:
                print(f"Epoch: {i + 1}/{n_epochs} | Loss: {loss.item():.4f} | Relative loss: {rel_loss:.4f}")

            if rel_loss < patience:
                print(f"Early Stopping at epoch: {i + 1}/{n_epochs} with loss: {loss.item():.4f}")
                break

            old_loss = loss
    except RuntimeError as e:
        print(data.train_x.dtype)
        raise e


def prepare_fold_data():
    train = df.drop(["medv"], axis=1).values
    targets = df["medv"].values


    x_train, x_test, y_train, y_test = train_test_split(train, targets, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return TrainingData(
        train_x=torch.from_numpy(x_train_scaled.astype(np.float32)),
        train_y=torch.from_numpy(y_train.astype(np.float32).reshape(-1, 1)),
        test_x=torch.from_numpy(x_test_scaled.astype(np.float32)),
        test_y=torch.from_numpy(y_test.astype(np.float32).reshape(-1, 1))
    )

@measure
def fold_training_loop(k: int = 4, n_epochs: int = 100, in_dims: int = 13):
    data = prepare_fold_data()
    train_data = data.train_x
    targets = data.train_y
    num_val_samples = len(train_data) // k


    fold_score = []

    for i in range(k):
        print(f"Processing fold {i}", end="")
        val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        val_targets = targets[i * num_val_samples:(i + 1) * num_val_samples]

        partial_train_data = torch.concatenate([
            train_data[:i * num_val_samples],
            train_data[(i + 1) * num_val_samples:]
        ], dim=0)

        partial_train_targets = torch.concatenate([
            targets[:i * num_val_samples],
            targets[(i + 1) * num_val_samples:]
        ])

        model = prepare_model(in_dims)

        for _ in range(n_epochs):
            model.optimizer.zero_grad()
            out = model.model(partial_train_data)

            loss = model.loss(out, partial_train_targets)
            loss.backward()
            model.optimizer.step()

        val_out = model.model(val_data)
        val_loss = model.loss(val_out, val_targets)
        fold_score.append(val_loss.item())

        print(f" | score: {val_loss.item():.4f}")

    avg_fold_score = np.mean(fold_score)
    print(f"Average Fold Score: {avg_fold_score:.4f}")
    return avg_fold_score


def main() -> None:
    no_fold_training_loop(n_epochs=500)
    print("---")
    fold_training_loop(k=5, n_epochs=500)


if __name__ == '__main__':
    main()
