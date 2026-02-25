#!/usr/bin/env python3
"""
Convert GFS GRIB2 wind data to JSON format for visualization.
Also saves a .npz grid file for direct use by tile generator (avoids JSON float issues).
"""

import xarray as xr  # type: ignore
import json
import numpy as np
import math
from scipy.interpolate import RegularGridInterpolator


def main():
    print("Loading GRIB2 file...")

    # Open the GRIB2 file
    ds = xr.open_dataset('gfs_wind.grib2', engine='cfgrib')

    print("Dataset variables:", list(ds.data_vars))
    print("Dataset coords:", list(ds.coords))
    print(ds)

    # Extract U and V wind components
    u10 = ds['u10'].values
    v10 = ds['v10'].values
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    print(f"\nGrid shape: {u10.shape}")
    print(f"Latitude range: {lats.min()} to {lats.max()}")
    print(f"Longitude range: {lons.min()} to {lons.max()}")

    # Interpolate to finer grid
    target_step = 0.0625
    target_lats = np.arange(lats.min(), lats.max() + target_step, target_step)
    target_lons = np.arange(lons.min(), lons.max() + target_step, target_step)

    print(f"Interpolating from {len(lats)}x{len(lons)} to {len(target_lats)}x{len(target_lons)}...")

    # RegularGridInterpolator expects ascending axes
    if lats[0] > lats[-1]:
        lats_asc = lats[::-1]
        u10_asc = u10[::-1, :]
        v10_asc = v10[::-1, :]
    else:
        lats_asc = lats
        u10_asc = u10
        v10_asc = v10

    interp_u = RegularGridInterpolator((lats_asc, lons), u10_asc, method='cubic',
                                        bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lats_asc, lons), v10_asc, method='cubic',
                                        bounds_error=False, fill_value=None)

    target_lat_mesh, target_lon_mesh = np.meshgrid(target_lats, target_lons, indexing='ij')
    points = np.column_stack([target_lat_mesh.ravel(), target_lon_mesh.ravel()])

    u_fine = interp_u(points).reshape(len(target_lats), len(target_lons))
    v_fine = interp_v(points).reshape(len(target_lats), len(target_lons))

    # Compute speed grid
    speed_grid = np.sqrt(u_fine**2 + v_fine**2).astype(np.float32)

    # Lats descending (90 to -90) for tile generator
    if target_lats[0] < target_lats[-1]:
        target_lats = target_lats[::-1]
        speed_grid = speed_grid[::-1, :]
        u_fine = u_fine[::-1, :]
        v_fine = v_fine[::-1, :]

    # Convert lons from 0-360 to -180 to 180 for the grid
    lon_converted = np.where(target_lons <= 180, target_lons, target_lons - 360)
    sort_idx = np.argsort(lon_converted)
    lon_converted = lon_converted[sort_idx]
    speed_grid = speed_grid[:, sort_idx]
    u_fine = u_fine[:, sort_idx]
    v_fine = v_fine[:, sort_idx]

    # Save as numpy grid (exact floats, no rounding issues)
    npz_file = 'data/wind_grid.npz'
    np.savez(npz_file,
             lats=target_lats,
             lons=lon_converted,
             speed=speed_grid,
             u=u_fine.astype(np.float32),
             v=v_fine.astype(np.float32))
    print(f"\nSaved grid to {npz_file}")
    print(f"Grid shape: {speed_grid.shape}")
    print(f"Lat range: {target_lats[0]} to {target_lats[-1]}")
    print(f"Lon range: {lon_converted[0]} to {lon_converted[-1]}")
    print(f"Speed range: {speed_grid.min():.2f} to {speed_grid.max():.2f} m/s")

    # Also save JSON for tooltip lookups (this can be lossy, it's just for display)
    print(f"\nBuilding JSON for tooltip lookups...")
    wind_data = []
    # Subsample for JSON - full grid is too big, use every 8th point
    step = 8
    for i in range(0, len(target_lats), step):
        for j in range(0, len(lon_converted), step):
            lat = float(target_lats[i])
            lon = float(lon_converted[j])
            spd = float(speed_grid[i, j])
            u = float(u_fine[i, j])
            v = float(v_fine[i, j])
            direction = (math.degrees(math.atan2(-u, -v)) + 360) % 360

            wind_data.append({
                'lat': round(lat, 2),
                'lon': round(lon, 2),
                'u': round(u, 2),
                'v': round(v, 2),
                'speed': round(spd, 2),
                'direction': round(direction, 1)
            })

    output_file = 'data/wind_data.json'
    with open(output_file, 'w') as f:
        json.dump(wind_data, f)

    print(f"Saved {len(wind_data):,} points to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    import os
    main()