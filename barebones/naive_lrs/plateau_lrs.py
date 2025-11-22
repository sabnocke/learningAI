from naive_layers import abstract


class ReduceLROnPlateau(interface.BaseLearningRateScheduler[float]):
    def __init__(self, init_lr, factor=0.1, patience=5, min_delta: float = 1e-4):
        self.init_lr = init_lr
        self.factor = factor
        self.patience = patience
        self.best_model = float('-inf')
        self.num_bad_epochs = 0
        self.current_lr = init_lr
        self.min_delta = min_delta
        self.old_change: None | float = None

    def step(self, n: float):
        if n < self.best_model - self.min_delta:
            self.best_model = n
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0
            self.current_lr *= self.factor

        return self.current_lr

    def rel_change(self, metric: float):
        if self.old_change is None:
            self.old_change = metric
            return False

        change = abs(metric - self.old_change) / self.old_change
        self.old_change = metric
        return change < self.min_delta