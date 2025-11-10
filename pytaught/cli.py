import argparse
from typing import List

from .data import CSVDataset, load_or_generate_data
from .metrics import mse_loss, r2_score
from .models import LinearReg, MLP, RandomForestRegressor
from .optim import SGD
from .preprocessing import StandardScaler
from .trainer import Trainer
from .utils import set_seed
from .version import __version__

__all__ = ["parse_args", "build_model_and_optimizer", "main"]


def build_model_and_optimizer(model_type: str, input_dim: int, hidden_sizes: List[int], lr: float, seed: int):
    if model_type == "linear":
        model = LinearReg(input_dim=input_dim, output_dim=1, seed=seed)
        params = [("W", model.W), ("b", model.b)]
    elif model_type == "mlp":
        model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=1, seed=seed)
        params = []
        for i, (W, b) in enumerate(zip(model.weights, model.biases)):
            params.append((f"W{i}", W))
            params.append((f"b{i}", b))
    else:
        raise ValueError(f"Unsupported model type for optimizer build: {model_type}")
    optim = SGD(params, lr=lr)
    return model, optim


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny ML/DL model on a CSV with pytaught")
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (numeric columns). If not given, synthetic data is generated.",
    )
    p.add_argument("--target", type=str, default="target", help="Name of the target column")
    p.add_argument("--model", choices=["linear", "mlp", "random_forest"], default="mlp")
    p.add_argument("--hidden-sizes", nargs="*", type=int, default=[64, 32], help="Hidden layer sizes for MLP")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="checkpoints/model.npz", help="Path prefix to save model (npz)")
    p.add_argument("--no-save", action="store_true", help="Do not save model")
    p.add_argument("--version", action="store_true", help="Print pytaught version and exit")
    p.add_argument("--rf-estimators", type=int, default=50, help="Number of trees for Random Forest")
    p.add_argument("--rf-max-depth", type=int, default=None, help="Maximum depth of each Random Forest tree")
    p.add_argument("--rf-min-samples-split", type=int, default=2, help="Minimum samples to split an internal node")
    p.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt",
        help='Feature subset strategy for Random Forest (int, float, "sqrt", "log2", "all")',
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.version:
        print(f"pytaught {__version__}")
        return
    set_seed(args.seed)

    # load/generate data
    df = load_or_generate_data(args.csv, args.target, generate_if_missing=True, seed=args.seed)
    dataset = CSVDataset(df=df, target=args.target)
    X_train_raw, y_train, X_val_raw, y_val = dataset.train_val_split(val_fraction=args.val_frac, shuffle=True, seed=args.seed)

    input_dim = X_train_raw.shape[1]
    print(f"Dataset: train {X_train_raw.shape[0]} rows, val {X_val_raw.shape[0]} rows, input_dim {input_dim}")

    if args.model == "random_forest":
        model = RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_split=args.rf_min_samples_split,
            max_features=args.rf_max_features,
            seed=args.seed,
        )
        print(
            f"Starting training: model=random_forest, estimators={args.rf_estimators}, "
            f"max_depth={args.rf_max_depth}, min_samples_split={args.rf_min_samples_split}, "
            f"max_features={args.rf_max_features}"
        )
        model.fit(X_train_raw, y_train.ravel())
        train_pred = model.forward(X_train_raw)
        val_pred = model.forward(X_val_raw)
        train_loss = mse_loss(y_train, train_pred)[0]
        val_loss = mse_loss(y_val, val_pred)[0]
        print(f"Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")
        train_metrics = {"mse": float(train_loss), "r2": float(r2_score(y_train, train_pred))}
        val_metrics = {"mse": float(val_loss), "r2": float(r2_score(y_val, val_pred))}
        print("Final Train:", train_metrics)
        print("Final Val  :", val_metrics)
        sample_pred = model.forward(X_val_raw[:5])
        print("Sample preds (first 5 val):")
        print(sample_pred.flatten())
        if not args.no_save:
            print("Warning: Saving Random Forest models is not supported in this minimal implementation.")
        return

    # scale for gradient-based models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    model, optim = build_model_and_optimizer(args.model, input_dim, args.hidden_sizes, args.lr, args.seed)

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
        save_path=save_path,
    )

    print(f"Starting training: model={args.model}, lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}")
    trainer.train()

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


if __name__ == "__main__":  # pragma: no cover
    main()

