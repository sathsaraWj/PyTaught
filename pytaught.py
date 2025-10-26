#!/usr/bin/env python3
"""
pytaught.py

Minimal ML/DL framework (single-file). Usage:

python pytaught.py --csv data.csv --target price --model mlp --hidden-sizes 64 32 --epochs 100

If you don't pass --csv, the script will generate a synthetic dataset and train on it.
"""


import argparse
import numpy as np
import pandas as pd
import os
import json
from typing import List, Tuple
from datetime import datetime

# -------------------------
# Utilities & Data classes
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)

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
        self.mean_ = arr['mean']
        self.scale_ = arr['scale']

class CSVDataset:
    def __init__(self, csv_path: str = None, target: str = None, df: pd.DataFrame = None):
        """
        Provide either csv_path OR df. target is the column name for labels.
        The loader drops rows with missing target and numeric-casts columns where possible.
        Non-numeric columns are dropped (basic approach).
        """
        if df is None:
            assert csv_path is not None, "Either csv_path or df must be provided"
            self.df = pd.read_csv(csv_path)
        else:
            self.df = df.copy()

        if target is None:
            raise ValueError("target column name must be provided")

        if target not in self.df.columns:
            raise ValueError(f"target column '{target}' not found in CSV columns: {list(self.df.columns)}")

        # drop rows with missing target
        self.df = self.df.dropna(subset=[target]).reset_index(drop=True)

        # keep numeric columns only (basic)
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        if target not in numeric_df.columns:
            # try converting target to numeric
            numeric_df[target] = pd.to_numeric(self.df[target], errors='coerce')
            numeric_df = numeric_df.dropna(subset=[target])
        self.df_numeric = numeric_df

        # separate X and y
        self.target = target
        self.X = self.df_numeric.drop(columns=[target]).values.astype(float) if len(self.df_numeric.columns) > 1 else np.zeros((len(self.df_numeric), 0))
        self.y = self.df_numeric[target].values.reshape(-1, 1).astype(float)

    def train_val_split(self, val_fraction: float = 0.2, shuffle: bool = True, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.X)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
        split = int(n * (1 - val_fraction))
        train_idx = idx[:split]
        val_idx = idx[split:]
        return self.X[train_idx], self.y[train_idx], self.X[val_idx], self.y[val_idx]

# -------------------------
# Model implementations
# -------------------------
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    returns (scalar loss, dloss/dy_pred)
    """
    n = y_true.shape[0]
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    dloss = (2.0 / n) * diff
    return loss, dloss

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)

class LinearReg:
    def __init__(self, input_dim: int, output_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim), dtype=float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.W) + self.b  # shape (n, out)

    def parameters(self):
        return [('W', self.W), ('b', self.b)]

    def zero_grads(self, grads):
        for k in grads:
            grads[k][:] = 0.0

    def backward(self, X, dloss_dy):
        n = X.shape[0]
        dW = X.T.dot(dloss_dy)  # shape (in, out)
        db = np.sum(dloss_dy, axis=0, keepdims=True)  # shape (1, out)
        return {'W': dW, 'b': db}

class MLP:
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        # weights: list of matrices W (in, out), biases b (1, out)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            # He init for ReLU hidden; small for output
            scale = np.sqrt(2.0 / max(1, in_dim))
            W = rng.randn(in_dim, out_dim) * scale
            b = np.zeros((1, out_dim), dtype=float)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Returns output and cache for backward.
        cache is list of dicts for each layer containing inputs and pre-activation z
        """
        a = X
        cache = []
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            z = a.dot(W) + b  # (n, out)
            is_last = (i == len(self.weights) - 1)
            if not is_last:
                a_next = np.maximum(0, z)  # ReLU
            else:
                a_next = z  # linear output
            cache.append({'a_in': a, 'z': z, 'W': W, 'b': b})
            a = a_next
        return a, cache

    def parameters(self):
        params = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params.append((f'W{i}', W))
            params.append((f'b{i}', b))
        return params

    def backward(self, dloss_dy: np.ndarray, cache: List[dict]) -> dict:
        """
        Backprop through cached forward computations.
        Returns dict mapping parameter names to gradients (same shapes as params).
        """
        grads = {}
        grad = dloss_dy  # gradient wrt network output
        L = len(cache)
        for i in reversed(range(L)):
            info = cache[i]
            a_in = info['a_in']  # shape (n, in)
            z = info['z']        # shape (n, out)
            W = info['W']        # shape (in, out)
            # Determine derivative through activation
            is_last = (i == L - 1)
            if not is_last:
                dz = grad * (z > 0).astype(float)  # ReLU'
            else:
                dz = grad  # linear
            # gradients
            dW = a_in.T.dot(dz)  # (in, out)
            db = np.sum(dz, axis=0, keepdims=True)  # (1, out)
            # gradient for previous layer's activation
            grad = dz.dot(W.T)  # (n, in)
            grads[f'W{i}'] = dW
            grads[f'b{i}'] = db
        return grads

# -------------------------
# Optimizer
# -------------------------
class SGD:
    def __init__(self, params_and_refs: List[Tuple[str, np.ndarray]], lr: float = 1e-3, weight_decay: float = 0.0):
        """
        params_and_refs: list of (name, numpy array reference) tuples.
        The optimizer will update arrays in-place using gradients passed to step().
        """
        self.params = params_and_refs
        self.lr = lr
        self.wd = weight_decay

    def step(self, grads: dict):
        for name, arr in self.params:
            grad = grads.get(name)
            if grad is None:
                continue
            # weight decay (L2)
            if self.wd != 0.0:
                arr -= self.lr * self.wd * arr
            arr -= self.lr * grad

# -------------------------
# Trainer
# -------------------------
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
        save_path: str = None
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
            batch_idx = idx[i:i+batch_size]
            yield X[batch_idx], y[batch_idx]

    def train(self):
        best_val = float('inf')
        patience = 0
        hist = {'train_loss': [], 'val_loss': []}
        for epoch in range(1, self.epochs + 1):
            # training
            train_losses = []
            for Xb, yb in self._iterate_minibatches(self.X_train, self.y_train, self.batch_size, shuffle=True):
                y_pred, cache = self.model.forward(Xb) if hasattr(self.model, 'forward') else (self.model.forward(Xb), None)
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
            train_pred = self.model.forward(self.X_train)[0] if isinstance(self.model, MLP) else self.model.forward(self.X_train)
            train_loss = mse_loss(self.y_train, train_pred)[0]
            val_pred = self.model.forward(self.X_val)[0] if isinstance(self.model, MLP) else self.model.forward(self.X_val)
            val_loss = mse_loss(self.y_val, val_pred)[0]
            hist['train_loss'].append(train_loss)
            hist['val_loss'].append(val_loss)

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

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.model.forward(X)[0] if isinstance(self.model, MLP) else self.model.forward(X)
        loss = mse_loss(y, y_pred)[0]
        r2 = r2_score(y, y_pred)
        return {'mse': float(loss), 'r2': float(r2)}

    def save(self, path: str):
        # Save parameters and scaler
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}
        if isinstance(self.model, LinearReg):
            data['W'] = self.model.W
            data['b'] = self.model.b
        else:
            for i, (W, b) in enumerate(zip(self.model.weights, self.model.biases)):
                data[f'W{i}'] = W
                data[f'b{i}'] = b
        np.savez(path, **data)
        # scaler
        if self.scaler:
            self.scaler.save(path + '.scaler.npz')
        print(f"Model saved to {path}")

    def load(self, path: str):
        arr = np.load(path)
        if isinstance(self.model, LinearReg):
            self.model.W = arr['W']
            self.model.b = arr['b']
        else:
            for i in range(len(self.model.weights)):
                self.model.weights[i] = arr[f'W{i}']
                self.model.biases[i] = arr[f'b{i}']
        if self.scaler:
            self.scaler.load(path + '.scaler.npz')
        print(f"Model loaded from {path}")

# -------------------------
# CLI & Example run
# -------------------------
def build_model_and_optimizer(model_type: str, input_dim: int, hidden_sizes: List[int], lr: float, seed: int):
    if model_type == 'linear':
        model = LinearReg(input_dim=input_dim, output_dim=1, seed=seed)
        params = [('W', model.W), ('b', model.b)]
    else:
        model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=1, seed=seed)
        # flatten params (W0, b0, W1, b1...)
        params = []
        for i, (W, b) in enumerate(zip(model.weights, model.biases)):
            params.append((f'W{i}', W))
            params.append((f'b{i}', b))
    optim = SGD(params, lr=lr)
    return model, optim

def load_or_generate_data(csv_path, target, generate_if_missing=True, n_samples=500, n_features=8, seed=42):
    if csv_path is None:
        # synthesize
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features)
        true_w = rng.randn(n_features, 1)
        y = X.dot(true_w) + 0.5 * rng.randn(n_samples, 1)
        df = pd.DataFrame(X, columns=[f'feat{i}' for i in range(n_features)])
        df[target] = y
        return df
    else:
        if not os.path.exists(csv_path):
            if generate_if_missing:
                print(f"CSV not found at '{csv_path}', generating synthetic data instead.")
                return load_or_generate_data(None, target, generate_if_missing, n_samples, n_features, seed)
            else:
                raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        return df

def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny ML/DL model on a CSV with myframework")
    p.add_argument('--csv', type=str, default=None, help='Path to CSV file (numeric columns). If not given, synthetic data is generated.')
    p.add_argument('--target', type=str, default='target', help='Name of the target column')
    p.add_argument('--model', choices=['linear', 'mlp'], default='mlp')
    p.add_argument('--hidden-sizes', nargs='*', type=int, default=[64, 32], help='Hidden layer sizes for MLP')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', type=str, default='checkpoints/model.npz', help='Path prefix to save model (npz)')
    p.add_argument('--no-save', action='store_true', help='Do not save model')
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    # load/generate data
    df = load_or_generate_data(args.csv, args.target, generate_if_missing=True, seed=args.seed)
    dataset = CSVDataset(df=df, target=args.target)
    X_train, y_train, X_val, y_val = dataset.train_val_split(val_fraction=args.val_frac, shuffle=True, seed=args.seed)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    input_dim = X_train.shape[1]
    print(f"Dataset: train {X_train.shape[0]} rows, val {X_val.shape[0]} rows, input_dim {input_dim}")

    model, optim = build_model_and_optimizer(args.model, input_dim, args.hidden_sizes, args.lr, args.seed)

    # Hook optimizer up to model references already done in build_model_and_optimizer

    # Trainer
    save_path = None if args.no_save else args.save
    trainer = Trainer(
        model=model,
        optimizer=optim,
        scaler=scaler,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        early_stop_patience=15,
        save_path=save_path
    )

    print(f"Starting training: model={args.model}, lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}")
    history = trainer.train()

    # final evaluation
    train_metrics = trainer.evaluate(X_train, y_train)
    val_metrics = trainer.evaluate(X_val, y_val)
    print("Final Train:", train_metrics)
    print("Final Val  :", val_metrics)

    # print a small prediction sample
    sample_pred = model.forward(X_val[:5])[0] if isinstance(model, MLP) else model.forward(X_val[:5])
    print("Sample preds (first 5 val):")
    print(sample_pred.flatten())
    if save_path:
        print("Model and scaler saved to:", save_path)

if __name__ == '__main__':
    main()
