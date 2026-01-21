#!/usr/bin/env python3
"""
Convert GFS GRIB2 wind data to JSON format for visualization.
"""

import xarray as xr # type: ignore
import json
import numpy as np
import math

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
    
    # Convert to list of points with lat, lon, u, v, speed, direction
    wind_data = []
    
    # Sample every N points to reduce data size (0.25° grid is very dense)
    # Original: 721 x 1440 = 1,038,240 points
    # Sample every 4 = ~65,000 points (similar to 1° grid)
    step = 4
    
    for i, lat in enumerate(lats[::step]):
        for j, lon in enumerate(lons[::step]):
            u = float(u10[i * step, j * step])
            v = float(v10[i * step, j * step])
            
            # Calculate speed and direction
            speed = math.sqrt(u**2 + v**2)
            
            # Direction: meteorological convention (where wind comes FROM)
            # atan2 gives angle of where wind is going TO, so add 180°
            direction = (math.degrees(math.atan2(-u, -v)) + 360) % 360
            
            # Convert longitude from 0-360 to -180 to 180
            lon_converted = lon if lon <= 180 else lon - 360
            
            wind_data.append({
                'lat': float(lat),
                'lon': float(lon_converted),
                'u': round(u, 2),
                'v': round(v, 2),
                'speed': round(speed, 2),
                'direction': round(direction, 1)
            })
    
    print(f"\nTotal points: {len(wind_data)}")
    print(f"Sample point: {wind_data[1000]}")
    
    # Save to JSON
    output_file = 'wind_data.json'
    with open(output_file, 'w') as f:
        json.dump(wind_data, f)
    
    print(f"\nSaved to {output_file}")
    print(f"File size: {len(json.dumps(wind_data)) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    main()