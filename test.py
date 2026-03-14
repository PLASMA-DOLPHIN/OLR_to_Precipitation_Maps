# test.py

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import RainDataset
from models_0 import UNetGenerator
import config

def test():
    # Use 2025 for testing
    olr_folder = "data/processed_olr/2025"
    rain_folder = "data/processed_rain/2025"

    dataset = RainDataset(olr_folder, rain_folder)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    G = UNetGenerator().to(config.DEVICE)
    # Load the latest checkpoint, assuming G_epoch_100.pth
    checkpoint = torch.load("checkpoints/best.pth", map_location=config.DEVICE)
    G.load_state_dict(checkpoint["G_state_dict"])
    G.eval()

    total_squared_error = 0.0
    total_diff = 0.0
    total_points = 0

    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    with torch.no_grad():
        for olr, rain in loader:
            olr = olr.to(config.DEVICE)
            rain = rain.to(config.DEVICE)

            fake_rain = G(olr)

            diff = fake_rain - rain
            total_squared_error += torch.sum(diff ** 2).item()
            total_diff += torch.sum(diff).item()
            total_points += diff.numel()

            x = fake_rain.view(-1).double()
            y = rain.view(-1).double()
            sum_x += torch.sum(x).item()
            sum_y += torch.sum(y).item()
            sum_x2 += torch.sum(x * x).item()
            sum_y2 += torch.sum(y * y).item()
            sum_xy += torch.sum(x * y).item()

    avg_mse = total_squared_error / total_points
    mean_bias = total_diff / total_points

    numerator = (total_points * sum_xy) - (sum_x * sum_y)
    denom_x = (total_points * sum_x2) - (sum_x ** 2)
    denom_y = (total_points * sum_y2) - (sum_y ** 2)
    denominator = np.sqrt(denom_x * denom_y)
    corr_coeff = numerator / denominator if denominator > 0 else float("nan")

    print(f"Average MSE on test set: {avg_mse:.4f}")
    print(f"Correlation Coefficient (Pearson r): {corr_coeff:.4f}")
    print(f"Mean Bias (Pred - True): {mean_bias:.4f}")

if __name__ == "__main__":
    test()
