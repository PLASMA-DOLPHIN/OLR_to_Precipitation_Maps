"""Utility for cropping and downsampling a CHIRPS NetCDF dataset.

Takes an input file with dimensions (time, latitude, longitude) and
produces one .npz file per day after cropping and reducing resolution.

Usage example:
    python process_chirps.py \
        --input chirps-v2.0.2023.days_p05.nc \
        --output-dir cropped_npz \
        --lon-range -130 -30 \
        --lat-range -50 50 \
        --resolution 0.25 \
        --method mean

The script supports several reduction methods when decreasing resolution:
 - mean (default)
 - median
 - max
 - min
 - nearest (simple decimation)

"""
import argparse
import os
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description="Crop and downsample CHIRPS data")
    parser.add_argument("--input", required=True, help="Path to input .nc file")
    parser.add_argument("--output-dir", required=True, help="Directory for npz outputs")
    parser.add_argument(
        "--lon-range",
        nargs=2,
        type=float,
        default=[-130, -30],
        help="Longitude range (min max) to crop",
    )
    parser.add_argument(
        "--lat-range",
        nargs=2,
        type=float,
        default=[-50, 50],
        help="Latitude range (min max) to crop",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Target spatial resolution in degrees (e.g. 0.25)",
    )
    parser.add_argument(
        "--method",
        choices=["mean", "median", "max", "min", "nearest"],
        default="mean",
        help="Reduction method when downsampling",
    )
    return parser.parse_args()


def crop_and_downsample(ds, lon_range, lat_range, target_res, method):
    """Crop to given lon/lat and reduce resolution to target_res.

    Parameters
    ----------
    ds : xr.Dataset or DataArray
        Dataset containing at least "precip" variable with dims
        (time, latitude, longitude).
    lon_range : tuple
        (min_lon, max_lon) in degrees east.
    lat_range : tuple
        (min_lat, max_lat) in degrees north.
    target_res : float
        Desired grid spacing (degrees). Must be a multiple of original
        resolution (0.05 in CHIRPS v2.0) or larger.
    method : str
        Reduction method: mean, median, max, min, nearest.

    Returns
    -------
    xr.DataArray
        Cropped and resampled precipitation with same time coordinate.
    """
    # select spatial subset
    da = ds["precip"].sel(
        longitude=slice(lon_range[0], lon_range[1]),
        latitude=slice(lat_range[0], lat_range[1]),
    )

    # original resolutions
    lon0 = float(ds.longitude.diff("longitude").mean().item())
    lat0 = float(ds.latitude.diff("latitude").mean().item())

    if target_res < lon0 or target_res < lat0:
        raise ValueError("Target resolution must be >= original resolution ({}, {})".format(lon0, lat0))

    # compute reduction factor
    factor_lon = int(round(target_res / lon0))
    factor_lat = int(round(target_res / lat0))

    if factor_lon <= 0 or factor_lat <= 0:
        raise ValueError("Computed factors must be positive")

    if method == "nearest":
        # downsample by simple slicing
        da = da.isel(
            longitude=slice(None, None, factor_lon),
            latitude=slice(None, None, factor_lat),
        )
    else:
        reduction = getattr(np, method)
        da = da.coarsen(
            longitude=factor_lon,
            latitude=factor_lat,
            boundary="trim",
        ).reduce(reduction)

    return da


def save_as_npz(da, out_dir):
    """Iterate over time dimension and save each slice as npz."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in da.time.values:
        subset = da.sel(time=t)
        data = subset.values
        # create filename using ISO date
        fname = f"{np.datetime_as_string(t, unit='D')}.npz"
        dest = out_dir / fname
        # compress to save space
        np.savez_compressed(dest, precip=data)


def main():
    args = parse_args()
    ds = xr.open_dataset(args.input)
    cropped = crop_and_downsample(
        ds,
        lon_range=args.lon_range,
        lat_range=args.lat_range,
        target_res=args.resolution,
        method=args.method,
    )
    save_as_npz(cropped, args.output_dir)


if __name__ == "__main__":
    main()
