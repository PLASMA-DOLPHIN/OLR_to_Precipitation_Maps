# dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RainDataset(Dataset):
    def __init__(self, olr_folders, rain_folders):
        self.olr_folders = olr_folders if isinstance(olr_folders, list) else [olr_folders]
        self.rain_folders = rain_folders if isinstance(rain_folders, list) else [rain_folders]

        self.paired_files = []

        for olr_folder, rain_folder in zip(self.olr_folders, self.rain_folders):
            olr_files = os.listdir(olr_folder)
            rain_files = os.listdir(rain_folder)

            # Extract available rain dates
            rain_dates = set(
                f.replace("rain-", "").replace(".npz", "")
                for f in rain_files
            )

            for f in olr_files:
                if not f.startswith("olr-"):
                    continue

                date = f.replace("olr-", "").replace(".npz", "")

                if date in rain_dates:
                    self.paired_files.append((date, olr_folder, rain_folder))

        self.paired_files.sort()

        print(f"Found {len(self.paired_files)} matched dates.")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        date, olr_folder, rain_folder = self.paired_files[idx]

        olr_path = os.path.join(olr_folder, f"olr-{date}.npz")
        rain_path = os.path.join(rain_folder, f"rain-{date}.npz")

        olr = np.load(olr_path)["olr"]
        rain = np.load(rain_path)["rain"]

        # Normalize OLR
        olr = (olr - olr.mean()) / (olr.std() + 1e-6)

        olr = torch.tensor(olr, dtype=torch.float32).unsqueeze(0)
        rain = torch.tensor(rain, dtype=torch.float32).unsqueeze(0)

        return olr, rain