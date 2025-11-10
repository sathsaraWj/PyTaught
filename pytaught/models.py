from typing import List, Tuple, Dict

import numpy as np

__all__ = [
    "LinearReg",
    "MLP",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
]


class LinearReg:
    def __init__(self, input_dim: int, output_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim), dtype=float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.W) + self.b  # shape (n, out)

    def parameters(self) -> List[Tuple[str, np.ndarray]]:
        return [("W", self.W), ("b", self.b)]

    def zero_grads(self, grads: Dict[str, np.ndarray]):
        for k in grads:
            grads[k][:] = 0.0

    def backward(self, X, dloss_dy):
        n = X.shape[0]
        dW = X.T.dot(dloss_dy)  # shape (in, out)
        db = np.sum(dloss_dy, axis=0, keepdims=True)  # shape (1, out)
        return {"W": dW, "b": db}


class MLP:
    def __init__(self, input_dim: int, hidden_sizes: List[int], output_dim: int = 1, seed: int = 42):
        rng = np.random.RandomState(seed)
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            # He init for ReLU hidden; small for output
            scale = np.sqrt(2.0 / max(1, in_dim))
            W = rng.randn(in_dim, out_dim) * scale
            b = np.zeros((1, out_dim), dtype=float)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: np.ndarray):
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
            is_last = i == len(self.weights) - 1
            if not is_last:
                a_next = np.maximum(0, z)  # ReLU
            else:
                a_next = z  # linear output
            cache.append({"a_in": a, "z": z, "W": W, "b": b})
            a = a_next
        return a, cache

    def parameters(self) -> List[Tuple[str, np.ndarray]]:
        params = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params.append((f"W{i}", W))
            params.append((f"b{i}", b))
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
            a_in = info["a_in"]  # shape (n, in)
            z = info["z"]  # shape (n, out)
            W = info["W"]  # shape (in, out)
            # Determine derivative through activation
            is_last = i == L - 1
            if not is_last:
                dz = grad * (z > 0).astype(float)  # ReLU'
            else:
                dz = grad  # linear
            # gradients
            dW = a_in.T.dot(dz)  # (in, out)
            db = np.sum(dz, axis=0, keepdims=True)  # (1, out)
            # gradient for previous layer's activation
            grad = dz.dot(W.T)  # (n, in)
            grads[f"W{i}"] = dW
            grads[f"b{i}"] = db
        return grads


class DecisionTreeRegressor:
    class Node:
        __slots__ = ("feature", "threshold", "left", "right", "value")

        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        seed: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.max_features = max_features
        self.seed = seed
        self.root = None
        self.n_features_ = None
        self.rng = np.random.RandomState(seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0:
            raise ValueError("DecisionTreeRegressor requires at least one feature.")
        self.n_features_ = X.shape[1]
        y = y.astype(float).ravel()
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for x in X], dtype=float)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X).reshape(-1, 1)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        n_samples = X.shape[0]
        if (
            n_samples < self.min_samples_split
            or self._is_pure(y)
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return self.Node(value=float(np.mean(y)))

        feature_indices = self._sample_features()
        best_feat, best_thr, best_score = None, None, float("inf")
        best_split = None

        for feat in feature_indices:
            X_column = X[:, feat]
            # sort for efficient split evaluation
            sorted_idx = np.argsort(X_column)
            X_sorted = X_column[sorted_idx]
            y_sorted = y[sorted_idx]

            for i in range(1, n_samples):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                thr = (X_sorted[i] + X_sorted[i - 1]) / 2.0
                left_y = y_sorted[:i]
                right_y = y_sorted[i:]
                if left_y.size == 0 or right_y.size == 0:
                    continue
                score = self._variance(left_y) * left_y.size + self._variance(right_y) * right_y.size
                if score < best_score:
                    best_feat = feat
                    best_thr = thr
                    best_score = score
                    best_split = (sorted_idx[:i].copy(), sorted_idx[i:].copy())

        if best_feat is None:
            return self.Node(value=float(np.mean(y)))

        left_indices, right_indices = best_split
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(feature=best_feat, threshold=best_thr, left=left, right=right)

    def _sample_features(self):
        total = self.n_features_
        if isinstance(self.max_features, int):
            k = max(1, min(self.max_features, total))
        elif isinstance(self.max_features, float):
            k = max(1, min(total, int(total * self.max_features)))
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(total)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(total)))
        else:  # 'all'
            k = total
        k = max(1, min(k, total))
        return self.rng.choice(total, size=k, replace=False)

    @staticmethod
    def _variance(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        return float(np.var(y))

    @staticmethod
    def _is_pure(y: np.ndarray) -> bool:
        return np.allclose(y, y[0])

    def _predict_row(self, x: np.ndarray, node):
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = None,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        seed: int = 42,
    ):
        self.n_estimators = max(1, n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.seed = seed
        self.trees: List[DecisionTreeRegressor] = []
        self.rng = np.random.RandomState(seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.astype(float).ravel()
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            tree_seed = self.rng.randint(0, 1_000_000)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=tree_seed,
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("RandomForestRegressor has not been fitted yet.")
        preds = np.stack([tree.predict(X) for tree in self.trees], axis=0)
        return preds.mean(axis=0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X).reshape(-1, 1)

