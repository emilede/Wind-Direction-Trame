#!/usr/bin/env python3
"""
Generate tile pyramid for wind visualization.
Uses VTK-style rendering pipeline, outputs tiles for Leaflet display.
"""

import json
import sys
import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image

# ============ CONFIG ============
TILE_SIZE = 256
MIN_ZOOM = 1
MAX_ZOOM = 5        # zoom 5 = 32x32 tiles = 8192x8192 effective
BLUR_SCALE = 3      # blur sigma = BLUR_SCALE * tile_resolution_factor
OUTPUT_DIR = 'data/tiles'
# ================================

def log(msg):
    print(msg)
    sys.stdout.flush()

def create_colormap():
    """Turbo colormap"""
    turbo = [
        (0.190, 0.072, 0.231), (0.217, 0.110, 0.352), (0.241, 0.150, 0.466),
        (0.259, 0.195, 0.568), (0.270, 0.245, 0.660), (0.275, 0.300, 0.740),
        (0.272, 0.360, 0.805), (0.262, 0.423, 0.856), (0.245, 0.489, 0.894),
        (0.222, 0.556, 0.919), (0.195, 0.622, 0.932), (0.167, 0.685, 0.933),
        (0.142, 0.743, 0.922), (0.127, 0.794, 0.899), (0.125, 0.838, 0.866),
        (0.142, 0.873, 0.823), (0.177, 0.901, 0.771), (0.229, 0.922, 0.710),
        (0.296, 0.937, 0.643), (0.375, 0.946, 0.571), (0.461, 0.949, 0.496),
        (0.550, 0.946, 0.420), (0.638, 0.937, 0.346), (0.720, 0.922, 0.278),
        (0.795, 0.901, 0.218), (0.860, 0.873, 0.170), (0.913, 0.838, 0.137),
        (0.954, 0.795, 0.120), (0.980, 0.745, 0.118), (0.993, 0.689, 0.127),
        (0.992, 0.627, 0.143), (0.978, 0.562, 0.162), (0.952, 0.495, 0.178),
        (0.916, 0.428, 0.189), (0.870, 0.363, 0.192), (0.816, 0.301, 0.187),
        (0.756, 0.244, 0.174), (0.691, 0.193, 0.155), (0.624, 0.149, 0.132),
        (0.555, 0.111, 0.107), (0.486, 0.080, 0.082), (0.419, 0.056, 0.058),
    ]
    
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        idx = min(int(t * (len(turbo) - 1)), len(turbo) - 2)
        frac = t * (len(turbo) - 1) - idx
        
        r = turbo[idx][0] + frac * (turbo[idx + 1][0] - turbo[idx][0])
        g = turbo[idx][1] + frac * (turbo[idx + 1][1] - turbo[idx][1])
        b = turbo[idx][2] + frac * (turbo[idx + 1][2] - turbo[idx][2])
        
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return lut

def get_tile_bounds(z, x, y):
    """Get lat/lon bounds for a tile. Using simple equirectangular scheme."""
    n_tiles = 2 ** z
    
    # Longitude: -180 to 180
    lon_per_tile = 360.0 / n_tiles
    lon_min = -180 + x * lon_per_tile
    lon_max = lon_min + lon_per_tile
    
    # Latitude: 90 to -90 (top to bottom)
    lat_per_tile = 180.0 / n_tiles
    lat_max = 90 - y * lat_per_tile
    lat_min = lat_max - lat_per_tile
    
    return lon_min, lon_max, lat_min, lat_max

def render_tile(lons, lats, speeds, z, x, y, lut):
    """Render a single tile."""
    lon_min, lon_max, lat_min, lat_max = get_tile_bounds(z, x, y)
    
    # Add padding for interpolation (avoid edge artifacts)
    pad = (lon_max - lon_min) * 0.1
    
    # Filter source points to tile region + padding
    mask = (
        (lons >= lon_min - pad) & (lons <= lon_max + pad) &
        (lats >= lat_min - pad) & (lats <= lat_max + pad)
    )
    
    if mask.sum() < 4:
        # Not enough points, return empty tile
        return None
    
    tile_lons = lons[mask]
    tile_lats = lats[mask]
    tile_speeds = speeds[mask]
    
    # Create grid for this tile
    grid_lon = np.linspace(lon_min, lon_max, TILE_SIZE)
    grid_lat = np.linspace(lat_max, lat_min, TILE_SIZE)  # top to bottom
    lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate
    grid_speeds = griddata(
        (tile_lons, tile_lats), tile_speeds,
        (lon_mesh, lat_mesh),
        method='linear',
        fill_value=0
    )
    
    # Blur - more at lower zoom levels
    blur_sigma = BLUR_SCALE * (2 ** (MAX_ZOOM - z))
    if blur_sigma > 0.5:
        grid_speeds = gaussian_filter(grid_speeds, sigma=blur_sigma)
    
    # Apply colormap
    normalized = np.clip(grid_speeds / 30.0, 0, 1)
    indices = (normalized * 255).astype(np.uint8)
    rgb = lut[indices]
    
    return Image.fromarray(rgb, mode='RGB')

def main():
    log("Loading wind data...")
    with open('data/wind_data.json') as f:
        wind_data = json.load(f)
    
    lons = np.array([p['lon'] for p in wind_data])
    lats = np.array([p['lat'] for p in wind_data])
    speeds = np.array([p['speed'] for p in wind_data])
    log(f"Loaded {len(wind_data):,} points")
    
    lut = create_colormap()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_tiles = 0
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        total_tiles += 4 ** z
    
    tile_count = 0
    
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n_tiles = 2 ** z
        log(f"\nZoom {z}: {n_tiles}x{n_tiles} tiles")
        
        zoom_dir = os.path.join(OUTPUT_DIR, str(z))
        os.makedirs(zoom_dir, exist_ok=True)
        
        for x in range(n_tiles):
            x_dir = os.path.join(zoom_dir, str(x))
            os.makedirs(x_dir, exist_ok=True)
            
            for y in range(n_tiles):
                tile_count += 1
                
                tile_path = os.path.join(x_dir, f"{y}.png")
                
                img = render_tile(lons, lats, speeds, z, x, y, lut)
                
                if img:
                    img.save(tile_path, optimize=True)
                else:
                    # Save empty tile
                    empty = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (30, 30, 40))
                    empty.save(tile_path)
                
                if tile_count % 20 == 0:
                    log(f"  Progress: {tile_count}/{total_tiles} tiles")
    
    log(f"\n✓ Done: {tile_count} tiles saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()