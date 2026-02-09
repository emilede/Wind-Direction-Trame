#!/usr/bin/env python3
"""
Convert GFS GRIB2 wind data to JSON format for visualization.
"""

import xarray as xr # type: ignore
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
    u10 = ds['u10'].values  # U-component of wind at 10m
    v10 = ds['v10'].values  # V-component of wind at 10m
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    
    print(f"\nGrid shape: {u10.shape}")
    print(f"Latitude range: {lats.min()} to {lats.max()}")
    print(f"Longitude range: {lons.min()} to {lons.max()}")
    
    # Interpolate to finer grid (0.125° spacing)
    target_step = 0.125
    target_lats = np.arange(lats.min(), lats.max() + target_step, target_step)
    target_lons = np.arange(lons.min(), lons.max() + target_step, target_step)
    
    print(f"Interpolating from {len(lats)}x{len(lons)} to {len(target_lats)}x{len(target_lons)}...")
    
    # RegularGridInterpolator expects ascending axes
    # GFS lats are typically descending (90 to -90), so flip
    if lats[0] > lats[-1]:
        lats_asc = lats[::-1]
        u10_asc = u10[::-1, :]
        v10_asc = v10[::-1, :]
    else:
        lats_asc = lats
        u10_asc = u10
        v10_asc = v10
    
    interp_u = RegularGridInterpolator((lats_asc, lons), u10_asc, method='cubic', bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((lats_asc, lons), v10_asc, method='cubic', bounds_error=False, fill_value=None)
    
    target_lat_mesh, target_lon_mesh = np.meshgrid(target_lats, target_lons, indexing='ij')
    points = np.column_stack([target_lat_mesh.ravel(), target_lon_mesh.ravel()])
    
    u_fine = interp_u(points)
    v_fine = interp_v(points)
    
    print(f"Interpolation done. Building JSON...")
    
    wind_data = []
    for i in range(len(points)):
        u = float(u_fine[i])
        v = float(v_fine[i])
        lat = float(points[i, 0])
        lon = float(points[i, 1])
        
        speed = math.sqrt(u**2 + v**2)
        direction = (math.degrees(math.atan2(-u, -v)) + 360) % 360
        
        lon_converted = lon if lon <= 180 else lon - 360
        
        wind_data.append({
            'lat': lat,
            'lon': round(lon_converted, 3),
            'u': round(u, 2),
            'v': round(v, 2),
            'speed': round(speed, 2),
            'direction': round(direction, 1)
        })
    
    print(f"\nTotal points: {len(wind_data)}")
    print(f"Sample point: {wind_data[1000]}")
    
    # Save to JSON
    output_file = 'data/wind_data.json'
    with open(output_file, 'w') as f:
        json.dump(wind_data, f)
    
    print(f"\nSaved to {output_file}")
    print(f"File size: {len(json.dumps(wind_data)) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()