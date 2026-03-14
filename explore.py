# describe_lnx.py
import numpy as np
from datetime import datetime, timedelta
import xarray as xr

def infer_dimensions(total_points):
    """
    Try to infer a sensible (ntime, nlat, nlon) triple from the length of the
    array by testing a handful of common climate‑model grids.
    """
    candidates = [
        (720, 1440),   # global 0.25° grid
        (721, 1440),   # with explicit poles
        (600, 1440),
        (400, 800),
        (360, 720),    # 0.5° grid
        (180, 360),    # 1° grid
    ]
    for nlat, nlon in candidates:
        if total_points % (nlat * nlon) == 0:
            ntime = total_points // (nlat * nlon)
            return int(ntime), nlat, nlon

    # fallback: brute‑force simple factorisation for a year of daily data
    for ntime in range(1, min(366, total_points // 1000)):
        remaining = total_points // ntime
        if ntime * remaining == total_points:
            for nlat in [180, 360, 400, 600, 720, 721]:
                if remaining % nlat == 0:
                    return ntime, nlat, remaining // nlat

    return None  # couldn't guess

def describe_lnx(path):
    print(f"reading {path!r}")
    data = np.fromfile(path, dtype='<f4')    # little‑endian float32
    npts = len(data)

    stats = {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std' : float(np.std(data)),
    }
    print("basic statistics")
    for k, v in stats.items():
        print(f"  {k:>4} = {v:.6g}")
    print(f"  total points = {npts:,}")

    dims = infer_dimensions(npts)
    if dims:
        ntime, nlat, nlon = dims
        print("\ninferred dimensions")
        print(f"  time steps : {ntime}")
        print(f"  latitudes  : {nlat}")
        print(f"  longitudes : {nlon}")
    else:
        print("\nunable to infer grid dimensions from length")

    # optionally construct an xarray dataset if dims were found
    if dims:
        lat = np.linspace(-90, 90, nlat)
        lon = np.linspace(-180, 180, nlon)
        base = datetime(2025, 1, 1)                     # adjust if known
        time = [base + timedelta(days=i) for i in range(ntime)]
        arr = data.reshape(ntime, nlat, nlon)
        ds = xr.Dataset({'variable': (('time','lat','lon'), arr)},
                        coords={'time': time, 'lat': lat, 'lon': lon})
        print("\nconstructed xarray Dataset")
        print(ds)
        return ds      # caller can inspect further
    return None

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("usage: python describe_lnx.py <file.lnx>")
        sys.exit(1)
    describe_lnx(sys.argv[1])