"""
Post-process cGAN outputs with non-linear regression.

Train on 2023-2024 GAN outputs vs. true rainfall, then evaluate on 2025.
Reports RMSE and Pearson correlation before and after post-processing.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from dataset import RainDataset
from models_0 import UNetGenerator
import config


class MetricAccumulator:
    def __init__(self) -> None:
        self.total_points = 0
        self.total_squared_error = 0.0
        self.total_diff = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = pred.astype(np.float64, copy=False)
        true = true.astype(np.float64, copy=False)

        self.total_points += pred.size
        diff = pred - true
        self.total_squared_error += float(np.sum(diff * diff))
        self.total_diff += float(np.sum(diff))

        self.sum_x += float(np.sum(pred))
        self.sum_y += float(np.sum(true))
        self.sum_x2 += float(np.sum(pred * pred))
        self.sum_y2 += float(np.sum(true * true))
        self.sum_xy += float(np.sum(pred * true))

    def finalize(self) -> Dict[str, float]:
        if self.total_points == 0:
            return {"rmse": float("nan"), "corr": float("nan"), "bias": float("nan")}

        mse = self.total_squared_error / self.total_points
        rmse = float(np.sqrt(mse))

        numerator = (self.total_points * self.sum_xy) - (self.sum_x * self.sum_y)
        denom_x = (self.total_points * self.sum_x2) - (self.sum_x ** 2)
        denom_y = (self.total_points * self.sum_y2) - (self.sum_y ** 2)
        denominator = np.sqrt(denom_x * denom_y)
        corr = float(numerator / denominator) if denominator > 0 else float("nan")
        bias = float(self.total_diff / self.total_points)

        return {"rmse": rmse, "corr": corr, "bias": bias}


def parse_years(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_folders(root: str, years: List[str]) -> List[str]:
    return [os.path.join(root, y) for y in years]


def resolve_checkpoint(path: str) -> str:
    candidates = []
    if path:
        candidates.append(path)
    if hasattr(config, "CHECKPOINT_DIR"):
        candidates.append(os.path.join(config.CHECKPOINT_DIR, "best.pth"))
        candidates.append(os.path.join(config.CHECKPOINT_DIR, "latest.pth"))
    candidates.extend([
        os.path.join("checkpoints", "best.pth"),
        os.path.join("checkpoints", "latest.pth"),
    ])

    for c in candidates:
        if c and os.path.exists(c):
            return c

    raise FileNotFoundError(
        "Could not find generator checkpoint. Provide --checkpoint path."
    )


def get_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return config.DEVICE


def clamp_array(arr: np.ndarray, clamp_min: float, clamp_max: float) -> np.ndarray:
    if clamp_min is None and clamp_max is None:
        return arr
    return np.clip(arr, clamp_min, clamp_max)


def collect_training_pairs(
    loader: DataLoader,
    generator: UNetGenerator,
    device: torch.device,
    max_points: int,
    clamp_min: float,
    clamp_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_list = []
    truth_list = []
    total = 0

    with torch.no_grad():
        for batch_idx, (olr, rain) in enumerate(loader):
            olr = olr.to(device)
            rain = rain.to(device)

            fake = generator(olr)

            fake_np = fake.detach().cpu().numpy().reshape(-1)
            truth_np = rain.detach().cpu().numpy().reshape(-1)

            fake_np = clamp_array(fake_np, clamp_min, clamp_max)
            truth_np = clamp_array(truth_np, clamp_min, clamp_max)

            if max_points is not None:
                remaining = max_points - total
                if remaining <= 0:
                    break
                if fake_np.size > remaining:
                    idx = np.random.choice(fake_np.size, size=remaining, replace=False)
                    fake_np = fake_np[idx]
                    truth_np = truth_np[idx]

            preds_list.append(fake_np)
            truth_list.append(truth_np)
            total += fake_np.size

            if (batch_idx + 1) % 10 == 0:
                print(f"  Collected {total:,} samples so far")

    X = np.concatenate(preds_list).reshape(-1, 1)
    y = np.concatenate(truth_list)
    return X, y


def build_models() -> Dict[str, Pipeline]:
    models = {
        "poly2_ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "hgb": Pipeline([
            ("scaler", StandardScaler()),
            ("hgb", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, max_depth=8, random_state=42
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=150, max_depth=20, random_state=42, n_jobs=-1
            )),
        ]),
        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=50.0, gamma="scale")),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=250,
                random_state=42,
                early_stopping=True
            )),
        ]),
    }
    return models


def evaluate_on_loader(
    loader: DataLoader,
    generator: UNetGenerator,
    device: torch.device,
    models: Dict[str, Pipeline],
    clamp_min: float,
    clamp_max: float,
) -> Dict[str, Dict[str, float]]:
    accumulators: Dict[str, MetricAccumulator] = {"gan": MetricAccumulator()}
    for name in models.keys():
        accumulators[name] = MetricAccumulator()

    with torch.no_grad():
        for batch_idx, (olr, rain) in enumerate(loader):
            olr = olr.to(device)
            rain = rain.to(device)

            fake = generator(olr)

            fake_np = fake.detach().cpu().numpy().reshape(-1)
            truth_np = rain.detach().cpu().numpy().reshape(-1)

            fake_np = clamp_array(fake_np, clamp_min, clamp_max)
            truth_np = clamp_array(truth_np, clamp_min, clamp_max)

            accumulators["gan"].update(fake_np, truth_np)

            X_batch = fake_np.reshape(-1, 1)
            for name, model in models.items():
                pred = model.predict(X_batch)
                pred = clamp_array(pred, clamp_min, clamp_max)
                accumulators[name].update(pred, truth_np)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(loader)} batches")

    return {name: acc.finalize() for name, acc in accumulators.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train non-linear post-processing on GAN outputs and evaluate on 2025."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--train-years", default="2023,2024")
    parser.add_argument("--test-year", default="2025")
    parser.add_argument("--olr-root", default="data/processed_olr")
    parser.add_argument("--rain-root", default="data/processed_rain")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-train-points", type=int, default=200000)
    parser.add_argument("--models", default="poly2_ridge,hgb,mlp")
    parser.add_argument("--clamp-min", type=float, default=None)
    parser.add_argument("--clamp-max", type=float, default=None)
    parser.add_argument("--save-dir", default="postprocess_models")
    parser.add_argument("--output", default="postprocess_results.json")
    args = parser.parse_args()

    np.random.seed(42)

    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading generator checkpoint: {checkpoint_path}")

    # Build datasets
    train_years = parse_years(args.train_years)
    test_years = parse_years(args.test_year)

    train_olr = build_folders(args.olr_root, train_years)
    train_rain = build_folders(args.rain_root, train_years)
    test_olr = build_folders(args.olr_root, test_years)
    test_rain = build_folders(args.rain_root, test_years)

    train_dataset = RainDataset(train_olr, train_rain)
    test_dataset = RainDataset(test_olr, test_rain)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load generator
    generator = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["G_state_dict"])
    generator.eval()

    # Collect training pairs
    max_points = args.max_train_points if args.max_train_points > 0 else None
    print("\nCollecting training samples from GAN outputs...")
    X_train, y_train = collect_training_pairs(
        train_loader,
        generator,
        device,
        max_points,
        args.clamp_min,
        args.clamp_max,
    )
    print(f"Training samples: {X_train.shape[0]:,}")

    # Build and fit models
    registry = build_models()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = {name: registry[name] for name in requested if name in registry}
    missing = [name for name in requested if name not in registry]
    if missing:
        print(f"Warning: unknown models skipped: {', '.join(missing)}")
    if not models:
        raise ValueError("No valid models selected. Use --models.")

    print("\nFitting regression models...")
    train_metrics = {}
    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        acc = MetricAccumulator()
        acc.update(pred_train, y_train)
        train_metrics[name] = acc.finalize()

    # Evaluate on test year(s)
    print("\nEvaluating on test year(s)...")
    test_metrics = evaluate_on_loader(
        test_loader,
        generator,
        device,
        models,
        args.clamp_min,
        args.clamp_max,
    )

    # Report
    print("\nResults on test year(s):")
    print(f"{'Model':<20} {'RMSE':>12} {'Corr':>12} {'Bias':>12}")
    print("-" * 60)
    base = test_metrics["gan"]
    print(
        f"{'GAN (raw)':<20} "
        f"{base['rmse']:>12.6f} {base['corr']:>12.6f} {base['bias']:>12.6f}"
    )
    for name in models.keys():
        m = test_metrics[name]
        print(f"{name:<20} {m['rmse']:>12.6f} {m['corr']:>12.6f} {m['bias']:>12.6f}")

    # Save trained models
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for name, model in models.items():
            model_path = os.path.join(args.save_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
        with open(os.path.join(args.save_dir, "models.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "models": list(models.keys()),
                    "checkpoint": checkpoint_path,
                    "train_years": train_years,
                    "clamp_min": args.clamp_min,
                    "clamp_max": args.clamp_max,
                },
                f,
                indent=2,
            )

    # Save summary
    summary = {
        "checkpoint": checkpoint_path,
        "train_years": train_years,
        "test_years": test_years,
        "max_train_points": max_points,
        "models": list(models.keys()),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "save_dir": args.save_dir,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
