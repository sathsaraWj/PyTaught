from typing import List, Tuple

import numpy as np

__all__ = ["SGD"]


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

