import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: int):
    np.random.seed(seed)

