from typing import Optional, TYPE_CHECKING, List
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from barebones.naive_lrs.abstract import BaseLearningRateScheduler

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    momentum: float
    every_nth: int = 1
    first_n: int = 0
    weight_decay: float = 0.
    lrs: Optional["BaseLearningRateScheduler"] = None
    batch_shuffle: bool = False
    batch_augment: bool = False
    patience: float = 1e-3
    patience_n: int = -1

    def __str__(self):
        coll = ""
        for key, value in self.__dict__.items():
            coll += f"{key}: {str(value)}\n"

        return coll

@dataclass
class History:
    loss: List[float] = field(default_factory=list)
    central_loss : List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    gap: List[float] = field(default_factory=list)

    def __len__(self):
        return len(self.test_loss)

    def entry(self, loss, test_loss, test_acc, gap):
        self.loss.append(loss)
        self.test_loss.append(test_loss)
        self.test_acc.append(test_acc)
        self.gap.append(gap)

    def analyze(self, recent_window = 0):
        epochs = len(self)
        if epochs < 2:
            print("Not enough data to analyze.")
            return

        start_idx = 0 if recent_window == 0 else max(0, epochs - recent_window)
        x = np.arange(start_idx, epochs)

        print(f"--- Trend Analysis (Epochs {start_idx} to {epochs}) ---")

        # Helper function to calculate slope
        def get_slope(data_list: List[float], name: str):
            data = np.array(data_list[start_idx:])
            slope, _ = np.polyfit(x, data, 1)

            # Interpret the slope
            status = "STABLE"
            if abs(slope) > 1e-5:
                if name == "Accuracy":
                    status = "IMPROVING" if slope > 0 else "DEGRADING"
                else:  # Losses
                    status = "IMPROVING" if slope < 0 else "DEGRADING"

            print(f"{name}: Slope = {slope:.6f} -> {status}")
            return slope

        # 1. Analyze Training Loss
        train_slope = get_slope(self.loss, "Train Loss")

        # 2. Analyze Test Loss
        test_slope = get_slope(self.test_loss, "Test Loss ")

        # 3. Analyze Test Accuracy
        acc_slope = get_slope(self.test_acc, "Test Acc  ")

        # 4. Detect Overfitting (The Logic Check)
        # Condition: Train Loss goes DOWN, Test Loss goes UP
        if train_slope < -1e-5 and test_slope > 1e-5:
            print("\nWARNING: OVERFITTING DETECTED")
            print("Training loss is decreasing while Test loss is increasing.")

    def plot(self, save_path=None):
        """Visualizes the history."""
        x = np.arange(1, len(self) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Losses
        ax1.plot(x, self.loss, label='Train Loss', color='blue')
        ax1.plot(x, self.gap, label="Generalization gap", color="red", linestyle="--")
        ax1.plot(x, self.test_loss, label='Test Loss', color='orange')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epochs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot Accuracy
        ax2.plot(x, self.test_acc, label='Test Accuracy', color='green')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epochs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
