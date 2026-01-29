#!/usr/bin/env python3
"""
Generate tile pyramid for wind visualization.
Dual-layer approach: glow + detail with alpha transparency.
Uses mercantile for correct Web Mercator projection.
"""

import json
import sys
import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PIL import Image
import mercantile

# ============ CONFIG ============
TILE_SIZE = 256
RENDER_SCALE = 1    # No oversampling for testing
MIN_ZOOM = 2
MAX_ZOOM = 3        # Just 2-3 for testing
PADDING = 32
OUTPUT_DIR = 'data/tiles'

# Dual layer settings
GLOW_BLUR = 4
DETAIL_BLUR = 1
GLOW_OPACITY = 0.5
DETAIL_OPACITY = 0.9

# Nonlinear transform exponent
SPEED_GAMMA = 0.7
# ================================

def log(msg):
    print(msg)
    sys.stdout.flush()

def create_colormap_rgba():
    """Turbo colormap with alpha based on speed."""
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
    
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        idx = min(int(t * (len(turbo) - 1)), len(turbo) - 2)
        frac = t * (len(turbo) - 1) - idx
        
        r = turbo[idx][0] + frac * (turbo[idx + 1][0] - turbo[idx][0])
        g = turbo[idx][1] + frac * (turbo[idx + 1][1] - turbo[idx][1])
        b = turbo[idx][2] + frac * (turbo[idx + 1][2] - turbo[idx][2])
        
        # Alpha ramps up with speed (low speed = more transparent)
        alpha = min(1.0, t * 1.5)  # Boost alpha curve
        
        lut[i] = [int(r * 255), int(g * 255), int(b * 255), int(alpha * 255)]
    
    return lut


def render_tile(lons, lats, speeds, z, x, y, lut):
    """Render dual-layer tile with glow + detail using Web Mercator."""
    # Get tile bounds using mercantile
    bounds = mercantile.bounds(x, y, z)
    lon_min, lat_min, lon_max, lat_max = bounds.west, bounds.south, bounds.east, bounds.north
    
    # Add padding in degrees (approximate)
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    pad_frac = PADDING / TILE_SIZE
    lon_pad = lon_range * pad_frac
    lat_pad = lat_range * pad_frac
    
    lon_min_padded = lon_min - lon_pad
    lon_max_padded = lon_max + lon_pad
    lat_min_padded = lat_min - lat_pad
    lat_max_padded = lat_max + lat_pad
    
    # Copy and wrap antimeridian
    all_lons = lons.copy()
    all_lats = lats.copy()
    all_speeds = speeds.copy()
    
    east_mask = lons > 170
    if east_mask.any():
        all_lons = np.concatenate([all_lons, lons[east_mask] - 360])
        all_lats = np.concatenate([all_lats, lats[east_mask]])
        all_speeds = np.concatenate([all_speeds, speeds[east_mask]])
    
    west_mask = lons < -170
    if west_mask.any():
        all_lons = np.concatenate([all_lons, lons[west_mask] + 360])
        all_lats = np.concatenate([all_lats, lats[west_mask]])
        all_speeds = np.concatenate([all_speeds, speeds[west_mask]])
    
    # Filter to tile region
    mask = (
        (all_lons >= lon_min_padded) & (all_lons <= lon_max_padded) &
        (all_lats >= lat_min_padded) & (all_lats <= lat_max_padded)
    )
    
    if mask.sum() < 4:
        return None
    
    tile_lons = all_lons[mask]
    tile_lats = all_lats[mask]
    tile_speeds = all_speeds[mask]
    
    # Create output grid - for each pixel, compute its lat/lon
    render_size = (TILE_SIZE + 2 * PADDING) * RENDER_SCALE
    
    # Pixel coordinates (with padding)
    px = np.linspace(-PADDING, TILE_SIZE + PADDING, render_size)
    py = np.linspace(-PADDING, TILE_SIZE + PADDING, render_size)
    px_mesh, py_mesh = np.meshgrid(px, py)
    
    # Convert pixel to lat/lon using mercantile
    # Each pixel maps to a fractional tile coordinate
    tile_frac_x = x + px_mesh / TILE_SIZE
    tile_frac_y = y + py_mesh / TILE_SIZE
    
    # Convert tile fractions to lat/lon
    n = 2 ** z
    grid_lon = tile_frac_x / n * 360.0 - 180.0
    
    # Mercator Y to latitude
    merc_y = np.pi * (1 - 2 * tile_frac_y / n)
    grid_lat = np.degrees(np.arctan(np.sinh(merc_y)))
    
    # Interpolate wind speeds onto this grid
    grid_speeds = griddata(
        (tile_lons, tile_lats), tile_speeds,
        (grid_lon, grid_lat),
        method='linear',
        fill_value=0
    )
    
    # Nonlinear transform for punch
    normalized = np.clip(grid_speeds / 30.0, 0, 1)
    normalized = np.power(normalized, SPEED_GAMMA)
    
    # Scale blur by zoom level
    zoom_scale = 2 ** (MAX_ZOOM - z)
    
    # GLOW LAYER: heavy blur
    glow_speeds = gaussian_filter(normalized, sigma=GLOW_BLUR * zoom_scale * RENDER_SCALE)
    glow_indices = (glow_speeds * 255).astype(np.uint8)
    glow_rgba = lut[glow_indices].astype(np.float32)
    glow_rgba[:, :, 3] *= GLOW_OPACITY
    
    # DETAIL LAYER: light blur
    detail_speeds = gaussian_filter(normalized, sigma=DETAIL_BLUR * RENDER_SCALE)
    detail_indices = (detail_speeds * 255).astype(np.uint8)
    detail_rgba = lut[detail_indices].astype(np.float32)
    detail_rgba[:, :, 3] *= DETAIL_OPACITY
    
    # Composite: glow underneath, detail on top (alpha blending)
    detail_alpha = detail_rgba[:, :, 3:4] / 255.0
    composite = detail_rgba.copy()
    composite[:, :, :3] = detail_rgba[:, :, :3] * detail_alpha + glow_rgba[:, :, :3] * (1 - detail_alpha)
    composite[:, :, 3] = np.clip(detail_rgba[:, :, 3] + glow_rgba[:, :, 3] * (1 - detail_alpha[:, :, 0]), 0, 255)
    
    composite = composite.astype(np.uint8)
    
    # Crop padding
    pad_px = PADDING * RENDER_SCALE
    final_size = TILE_SIZE * RENDER_SCALE
    composite = composite[pad_px:pad_px+final_size, pad_px:pad_px+final_size]
    
    # Downsample to final size
    img = Image.fromarray(composite, 'RGBA')
    img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
    
    return img


def main():
    log("Loading wind data...")
    with open('data/wind_data.json') as f:
        wind_data = json.load(f)
    
    lons = np.array([p['lon'] for p in wind_data])
    lats = np.array([p['lat'] for p in wind_data])
    speeds = np.array([p['speed'] for p in wind_data])
    log(f"Loaded {len(wind_data):,} points")
    
    # Debug: show tile bounds
    log("\n=== Tile bounds (mercantile) ===")
    for ty in range(4):
        b = mercantile.bounds(0, ty, 2)
        log(f"Tile (2,0,{ty}): lat {b.south:.1f} to {b.north:.1f}")
    
    lut = create_colormap_rgba()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_tiles = sum(4 ** z for z in range(MIN_ZOOM, MAX_ZOOM + 1))
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
                    empty = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
                    empty.save(tile_path)
                
                if tile_count % 10 == 0:
                    log(f"  Progress: {tile_count}/{total_tiles} tiles")
    
    log(f"\n✓ Done: {tile_count} tiles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()