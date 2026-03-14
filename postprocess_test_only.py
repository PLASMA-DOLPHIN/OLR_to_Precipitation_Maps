"""
Test-only script for evaluating GAN outputs and trained post-processing models.

Loads trained regression models from a directory and reports RMSE, Pearson
correlation, and mean bias on the test year(s).
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import joblib

from dataset import RainDataset
from models import UNetGenerator
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
        os.path.join("checkpoints", "best.pth")
        # os.path.join("checkpoints", "latest.pth")
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


def filter_by_threshold(
    truth: np.ndarray,
    pred: np.ndarray,
    metric_min: float,
    metric_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if metric_min is None and metric_max is None:
        return truth, pred

    mask = np.ones_like(truth, dtype=bool)
    if metric_min is not None:
        mask &= truth >= metric_min
    if metric_max is not None:
        mask &= truth <= metric_max

    truth_f = truth[mask]
    pred_f = pred[mask]

    # Clamp both truth and pred to metric bounds (if provided)
    if metric_min is not None or metric_max is not None:
        truth_f = np.clip(truth_f, metric_min, metric_max)
        pred_f = np.clip(pred_f, metric_min, metric_max)

    return truth_f, pred_f


def load_models(save_dir: str, requested: List[str]) -> Dict[str, object]:
    models_json = os.path.join(save_dir, "models.json")
    model_names = []

    if os.path.exists(models_json):
        with open(models_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_names = data.get("models", [])
    else:
        for name in os.listdir(save_dir):
            if name.endswith(".joblib"):
                model_names.append(name.replace(".joblib", ""))

    if requested:
        model_names = [m for m in model_names if m in requested]

    models = {}
    for name in model_names:
        path = os.path.join(save_dir, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)

    if not models:
        raise ValueError("No models loaded. Check --models or --model-dir.")

    return models


def evaluate_on_loader(
    loader: DataLoader,
    generator: UNetGenerator,
    device: torch.device,
    models: Dict[str, object],
    clamp_min: float,
    clamp_max: float,
    metric_min: float,
    metric_max: float,
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

            truth_f, pred_f = filter_by_threshold(truth_np, fake_np, metric_min, metric_max)
            accumulators["gan"].update(pred_f, truth_f)

            X_batch = fake_np.reshape(-1, 1)
            for name, model in models.items():
                pred = model.predict(X_batch)
                pred = clamp_array(pred, clamp_min, clamp_max)
                truth_f, pred_f = filter_by_threshold(truth_np, pred, metric_min, metric_max)
                accumulators[name].update(pred_f, truth_f)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated {batch_idx + 1}/{len(loader)} batches")

    return {name: acc.finalize() for name, acc in accumulators.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-only metrics for GAN outputs and trained post-processing models."
    )
    parser.add_argument("--model-dir", default="postprocess_models")
    parser.add_argument("--models", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test-year", default="2025")
    parser.add_argument("--olr-root", default="data/processed_olr")
    parser.add_argument("--rain-root", default="data/processed_rain")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default="")
    parser.add_argument("--clamp-min", type=float, default=None)
    parser.add_argument("--clamp-max", type=float, default=None)
    parser.add_argument("--metric-min", type=float, default=None)
    parser.add_argument("--metric-max", type=float, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Pull defaults from models.json if present
    models_json = os.path.join(args.model_dir, "models.json")
    if os.path.exists(models_json):
        with open(models_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not args.checkpoint:
            args.checkpoint = data.get("checkpoint", "")
        if args.clamp_min is None:
            args.clamp_min = data.get("clamp_min", None)
        if args.clamp_max is None:
            args.clamp_max = data.get("clamp_max", None)

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    print(f"Loading generator checkpoint: {checkpoint_path}")

    # Load generator
    generator = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["G_state_dict"])
    generator.eval()

    # Load models
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models = load_models(args.model_dir, requested)

    # Build dataset
    test_years = parse_years(args.test_year)
    test_olr = build_folders(args.olr_root, test_years)
    test_rain = build_folders(args.rain_root, test_years)
    test_dataset = RainDataset(test_olr, test_rain)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Evaluate
    print("\nEvaluating on test year(s)...")
    test_metrics = evaluate_on_loader(
        test_loader,
        generator,
        device,
        models,
        args.clamp_min,
        args.clamp_max,
        args.metric_min,
        args.metric_max,
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


if __name__ == "__main__":
    main()
