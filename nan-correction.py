import numpy as np
import os
from datetime import datetime

# === INPUT FOLDERS ===
olr_folder = "/data/olr/2023"
rain_folder = "/data/precip/cropped_npz_chirps23"

# === OUTPUT FOLDERS ===
output_olr_folder = "data/processed_olr/2023"
output_rain_folder = "data/processed_rain/2023"

os.makedirs(output_olr_folder, exist_ok=True)
os.makedirs(output_rain_folder, exist_ok=True)

# === BUILD RAIN LOOKUP ===
rain_files = {
    f.replace(".npz", ""): f
    for f in os.listdir(rain_folder)
    if f.endswith(".npz")
}

processed = 0
skipped = 0

for olr_file in sorted(os.listdir(olr_folder)):

    if not olr_file.endswith(".npz"):
        continue

    # OLR format: OLR_Americas_20250101.npz
    date_str = olr_file.replace("OLR_Americas_", "").replace(".npz", "")

    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"Skipping malformed file: {olr_file}")
        skipped += 1
        continue

    rain_date = date_obj.strftime("%Y-%m-%d")

    if rain_date not in rain_files:
        print(f"Missing rain file for {rain_date}")
        skipped += 1
        continue

    rain_file = rain_files[rain_date]

    # === LOAD ===
    olr = np.load(os.path.join(olr_folder, olr_file))["olr"]
    rain = np.load(os.path.join(rain_folder, rain_file))["precip"]

    # === SHAPE CHECK ===
    if olr.shape != (400, 400) or rain.shape != (400, 400):
        print(f"Shape mismatch on {rain_date}")
        skipped += 1
        continue

    # === CREATE LAND MASK FROM RAIN ===
    land_mask = ~np.isnan(rain)

    # === HANDLE RAIN NaNs ===
    rain_clean = np.nan_to_num(rain, nan=0.0).astype(np.float32)

    # === REMOVE OCEAN FROM OLR ===
    olr_clean = np.where(land_mask, olr, 350).astype(np.float32)

    # === SAVE WITH SAME DATE FORMAT ===
    olr_output_name = f"olr-{rain_date}.npz"
    rain_output_name = f"rain-{rain_date}.npz"

    np.savez_compressed(
        os.path.join(output_olr_folder, olr_output_name),
        olr=olr_clean
    )

    np.savez_compressed(
        os.path.join(output_rain_folder, rain_output_name),
        rain=rain_clean
    )

    processed += 1

print("\nDone.")
print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
