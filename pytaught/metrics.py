import numpy as np

__all__ = ["mse_loss", "r2_score"]


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray):
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

