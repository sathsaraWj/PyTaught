import os
from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ["CSVDataset", "load_or_generate_data"]


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
            numeric_df[target] = pd.to_numeric(self.df[target], errors="coerce")
            numeric_df = numeric_df.dropna(subset=[target])
        self.df_numeric = numeric_df

        # separate X and y
        self.target = target
        self.X = (
            self.df_numeric.drop(columns=[target]).values.astype(float)
            if len(self.df_numeric.columns) > 1
            else np.zeros((len(self.df_numeric), 0))
        )
        self.y = self.df_numeric[target].values.reshape(-1, 1).astype(float)

    def train_val_split(
        self, val_fraction: float = 0.2, shuffle: bool = True, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(self.X)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(idx)
        split = int(n * (1 - val_fraction))
        train_idx = idx[:split]
        val_idx = idx[split:]
        return self.X[train_idx], self.y[train_idx], self.X[val_idx], self.y[val_idx]


def load_or_generate_data(
    csv_path,
    target,
    generate_if_missing=True,
    n_samples=500,
    n_features=8,
    seed=42,
):
    if csv_path is None:
        # synthesize
        rng = np.random.RandomState(seed)
        X = rng.randn(n_samples, n_features)
        true_w = rng.randn(n_features, 1)
        y = X.dot(true_w) + 0.5 * rng.randn(n_samples, 1)
        df = pd.DataFrame(X, columns=[f"feat{i}" for i in range(n_features)])
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

