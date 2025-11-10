import numpy as np

__all__ = ["StandardScaler"]


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        np.savez(path, mean=self.mean_, scale=self.scale_)

    def load(self, path: str):
        arr = np.load(path)
        self.mean_ = arr["mean"]
        self.scale_ = arr["scale"]

