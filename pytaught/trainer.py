import os
from typing import Dict

import numpy as np

from .metrics import mse_loss, r2_score
from .models import LinearReg, MLP
from .preprocessing import StandardScaler

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scaler: StandardScaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        seed: int = 42,
        early_stop_patience: int = 10,
        save_path: str = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.early_stop_patience = early_stop_patience
        self.save_path = save_path

    def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle=True):
        n = X.shape[0]
        idx = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i : i + batch_size]
            yield X[batch_idx], y[batch_idx]

    def train(self):
        best_val = float("inf")
        patience = 0
        hist = {"train_loss": [], "val_loss": []}
        for epoch in range(1, self.epochs + 1):
            # training
            train_losses = []
            for Xb, yb in self._iterate_minibatches(self.X_train, self.y_train, self.batch_size, shuffle=True):
                if isinstance(self.model, MLP):
                    y_pred, cache = self.model.forward(Xb)
                else:
                    y_pred = self.model.forward(Xb)
                    cache = None
                loss, dloss = mse_loss(yb, y_pred)
                train_losses.append(loss)
                # backward pass
                if isinstance(self.model, LinearReg):
                    grads = self.model.backward(Xb, dloss)
                else:  # MLP
                    grads = self.model.backward(dloss, cache)
                # optimizer step
                self.optimizer.step(grads)

            # epoch metrics
            train_pred = (
                self.model.forward(self.X_train)[0]
                if isinstance(self.model, MLP)
                else self.model.forward(self.X_train)
            )
            train_loss = mse_loss(self.y_train, train_pred)[0]
            val_pred = (
                self.model.forward(self.X_val)[0]
                if isinstance(self.model, MLP)
                else self.model.forward(self.X_val)
            )
            val_loss = mse_loss(self.y_val, val_pred)[0]
            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)

            print(f"[{epoch:03d}/{self.epochs}] Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")

            # early stopping
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                patience = 0
                # save model if requested
                if self.save_path:
                    self.save(self.save_path)
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs. Best val MSE: {best_val:.6f}")
                    break
        return hist

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.model.forward(X)[0] if isinstance(self.model, MLP) else self.model.forward(X)
        loss = mse_loss(y, y_pred)[0]
        r2 = r2_score(y, y_pred)
        return {"mse": float(loss), "r2": float(r2)}

    def save(self, path: str):
        # Save parameters and scaler
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        if isinstance(self.model, LinearReg):
            data["W"] = self.model.W
            data["b"] = self.model.b
        else:
            for i, (W, b) in enumerate(zip(self.model.weights, self.model.biases)):
                data[f"W{i}"] = W
                data[f"b{i}"] = b
        np.savez(path, **data)
        # scaler
        if self.scaler:
            self.scaler.save(path + ".scaler.npz")
        print(f"Model saved to {path}")

    def load(self, path: str):
        arr = np.load(path)
        if isinstance(self.model, LinearReg):
            self.model.W = arr["W"]
            self.model.b = arr["b"]
        else:
            for i in range(len(self.model.weights)):
                self.model.weights[i] = arr[f"W{i}"]
                self.model.biases[i] = arr[f"b{i}"]
        if self.scaler:
            self.scaler.load(path + ".scaler.npz")
        print(f"Model loaded from {path}")

