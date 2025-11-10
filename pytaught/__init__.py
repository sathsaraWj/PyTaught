"""
pytaught package exports.
"""

from .version import __version__
from .utils import set_seed
from .preprocessing import StandardScaler
from .data import CSVDataset, load_or_generate_data
from .metrics import mse_loss, r2_score
from .models import LinearReg, MLP, DecisionTreeRegressor, RandomForestRegressor
from .optim import SGD
from .trainer import Trainer

__all__ = [
    "__version__",
    "set_seed",
    "StandardScaler",
    "CSVDataset",
    "load_or_generate_data",
    "mse_loss",
    "r2_score",
    "LinearReg",
    "MLP",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "SGD",
    "Trainer",
]
