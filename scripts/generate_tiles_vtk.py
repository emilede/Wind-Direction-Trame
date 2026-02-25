#!/usr/bin/env python3
"""
Generate tile pyramid for wind visualization using VTK.

RENDERING APPROACH (for thesis documentation):
===============================================
Instead of Delaunay triangulation (which interpolates within triangles and
can leave visible edges at triangle boundaries), this pipeline uses per-pixel
interpolation from a regular grid:

1. Wind data (already a regular grid from convert.py) is reshaped directly
2. For each tile, pixel coordinates are converted to lat/lng via Web Mercator
3. Wind speed at each pixel is bilinearly interpolated from the regular grid
4. The speed values are packed into a vtkImageData structure
5. VTK's vtkImageMapToColors applies the colormap via a vtkLookupTable
6. The resulting RGB image is composited with terrain hillshade

This produces perfectly smooth, per-pixel color gradients with zero triangle
artifacts, matching the quality of professional weather visualization services.
===============================================
"""

import json
import sys
import os
import numpy as np
import mercantile
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PIL import Image
from scipy.ndimage import gaussian_filter
import urllib.request

# ============ CONFIG ============
TILE_SIZE = 256
RENDER_SCALE = 4
PADDING = 32
MIN_ZOOM = 3
MAX_ZOOM = 5
OUTPUT_DIR = os.path.abspath('data/tiles')
TERRAIN_CACHE = os.path.abspath('data/terrain_cache')

GLOW_BLUR = 1
DETAIL_BLUR = 0.3
# ================================

def log(msg):
    print(msg)
    sys.stdout.flush()


def fetch_terrain_tile(z, x, y):
    cache_path = os.path.join(TERRAIN_CACHE, str(z), str(x), f"{y}.png")
    if os.path.exists(cache_path):
        img = Image.open(cache_path).convert('L')
        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
        return np.array(img)

    url = f"https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade_Dark/MapServer/tile/{z}/{y}/{x}"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        img = Image.open(cache_path).convert('L')
        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        log(f"  Warning: terrain fetch failed for {z}/{x}/{y}: {e}")
        return np.full((TILE_SIZE, TILE_SIZE), 128, dtype=np.uint8)


def create_vtk_lut():
    """Create VTK lookup table with exact Zoom Earth colors (pixel-sampled)."""
    # Colors sampled directly from Zoom Earth at each mph value
    # Format: (t, (R, G, B)) where t = speed_ms / 35
    colors = [
        # 0 mph (0 m/s)
        (0.000, (91, 70, 168)),
        # 5 mph (2.24 m/s)
        (0.064, (87, 112, 195)),
        # 10 mph (4.47 m/s)
        (0.128, (95, 155, 207)),
        # 15 mph (6.71 m/s)
        (0.192, (113, 196, 201)),
        # 20 mph (8.94 m/s)
        (0.255, (133, 225, 174)),
        # 25 mph (11.18 m/s)
        (0.319, (176, 234, 154)),
        # 30 mph (13.41 m/s)
        (0.383, (228, 242, 145)),
        # 35 mph (15.65 m/s)
        (0.447, (245, 228, 134)),
        # 40 mph (17.88 m/s)
        (0.511, (248, 192, 123)),
        # 45 mph (20.12 m/s)
        (0.575, (243, 156, 112)),
        # 50 mph (22.35 m/s)
        (0.639, (231, 127, 105)),
        # 55 mph (24.59 m/s)
        (0.703, (216, 105, 99)),
        # 60 mph (26.82 m/s)
        (0.766, (196, 88, 99)),
        # 65 mph (29.06 m/s)
        (0.830, (174, 74, 103)),
        # 70 mph (31.29 m/s)
        (0.894, (155, 62, 106)),
        # 75 mph (33.53 m/s)
        (0.958, (138, 50, 105)),
        # 80 mph (35.76 m/s)
        (1.000, (122, 39, 106)),
    ]

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(1024)
    lut.SetRange(0, 35)

    for i in range(1024):
        t = i / 1023.0
        for j in range(len(colors) - 1):
            if colors[j][0] <= t <= colors[j+1][0]:
                t0, c0 = colors[j]
                t1, c1 = colors[j+1]
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = (c0[0] + frac * (c1[0] - c0[0])) / 255
                g = (c0[1] + frac * (c1[1] - c0[1])) / 255
                b = (c0[2] + frac * (c1[2] - c0[2])) / 255
                lut.SetTableValue(i, r, g, b, 1.0)
                break

    lut.Build()
    return lut


class TileRenderer:
    def __init__(self, tile_size, render_scale, padding, lut):
        self.tile_size = tile_size
        self.padding = padding
        self.render_scale = render_scale
        self.lut = lut
        self.padded_size = tile_size + 2 * padding
        self.render_size = self.padded_size * render_scale

    def render_wind_color(self, speed_grid, grid_lats, grid_lons, z, x, y):
        size = self.render_size
        bounds = mercantile.bounds(x, y, z)
        lon_min, lat_min, lon_max, lat_max = bounds.west, bounds.south, bounds.east, bounds.north

        lon_range = lon_max - lon_min
        pad_frac = self.padding / self.tile_size
        padded_lon_min = lon_min - lon_range * pad_frac
        padded_lon_max = lon_max + lon_range * pad_frac

        n = 2 ** z
        tile_merc_min = y / n
        tile_merc_max = (y + 1) / n
        merc_range = tile_merc_max - tile_merc_min
        padded_merc_min = tile_merc_min - merc_range * pad_frac
        padded_merc_max = tile_merc_max + merc_range * pad_frac

        pixel_lons = np.linspace(padded_lon_min, padded_lon_max, size)
        pixel_merc = np.linspace(padded_merc_min, padded_merc_max, size)
        pixel_lats = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * pixel_merc))))

        lat_step = abs(grid_lats[0] - grid_lats[1])
        lon_step = abs(grid_lons[1] - grid_lons[0])

        lat_frac_1d = (grid_lats[0] - pixel_lats) / lat_step
        lon_frac_1d = (pixel_lons - grid_lons[0]) / lon_step

        lon_frac_2d, lat_frac_2d = np.meshgrid(lon_frac_1d, lat_frac_1d)

        lat_0 = np.clip(np.floor(lat_frac_2d).astype(int), 0, speed_grid.shape[0] - 2)
        lon_0 = np.clip(np.floor(lon_frac_2d).astype(int), 0, speed_grid.shape[1] - 2)
        lat_f = np.clip(lat_frac_2d - lat_0, 0, 1)
        lon_f = np.clip(lon_frac_2d - lon_0, 0, 1)

        speeds = (
            speed_grid[lat_0, lon_0] * (1 - lat_f) * (1 - lon_f) +
            speed_grid[lat_0 + 1, lon_0] * lat_f * (1 - lon_f) +
            speed_grid[lat_0, lon_0 + 1] * (1 - lat_f) * lon_f +
            speed_grid[lat_0 + 1, lon_0 + 1] * lat_f * lon_f
        )

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(size, size, 1)
        image_data.SetSpacing(1.0, 1.0, 1.0)
        image_data.SetOrigin(0, 0, 0)

        speed_for_vtk = speeds[::-1, :].ravel().astype(np.float32)
        vtk_scalars = numpy_to_vtk(speed_for_vtk, deep=True)
        vtk_scalars.SetName("speed")
        image_data.GetPointData().SetScalars(vtk_scalars)

        color_map = vtk.vtkImageMapToColors()
        color_map.SetLookupTable(self.lut)
        color_map.SetOutputFormatToRGB()
        color_map.SetInputData(image_data)
        color_map.Update()

        output = color_map.GetOutput()
        rgb_vtk = output.GetPointData().GetScalars()
        arr = vtk_to_numpy(rgb_vtk).reshape(size, size, 3)
        arr = np.flipud(arr.copy())
        return arr

    def composite_tile(self, wind_rgb_padded, terrain_gray):
        glow = np.zeros_like(wind_rgb_padded, dtype=float)
        for c in range(3):
            glow[:, :, c] = gaussian_filter(wind_rgb_padded[:, :, c].astype(float), sigma=GLOW_BLUR)

        detail = np.zeros_like(wind_rgb_padded, dtype=float)
        for c in range(3):
            detail[:, :, c] = gaussian_filter(wind_rgb_padded[:, :, c].astype(float), sigma=DETAIL_BLUR)

        wind_combined = (glow * 0.2 + detail * 0.8)

        crop_start = self.padding * self.render_scale
        crop_end = crop_start + self.tile_size * self.render_scale
        wind_cropped = wind_combined[crop_start:crop_end, crop_start:crop_end]

        if terrain_gray.shape[0] != wind_cropped.shape[0]:
            terrain_img = Image.fromarray(terrain_gray)
            terrain_img = terrain_img.resize((wind_cropped.shape[1], wind_cropped.shape[0]), Image.LANCZOS)
            terrain_gray = np.array(terrain_img)

        terrain_norm = terrain_gray.astype(float) / 255.0
        terrain_norm = np.clip(terrain_norm * 2.0, 0, 1)

        result = np.zeros((wind_cropped.shape[0], wind_cropped.shape[1], 3), dtype=float)
        for c in range(3):
            w = wind_cropped[:, :, c] / 255.0
            t = terrain_norm
            result[:, :, c] = 1 - (1 - t * 0.1) * (1 - w)

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result


def main():
    log("Loading wind data...")
    with open('data/wind_data.json') as f:
        wind_data = json.load(f)

    n_points = len(wind_data)
    log(f"Loaded {n_points:,} points")

    speeds = np.array([p['speed'] for p in wind_data], dtype=np.float32)

    # Figure out grid dimensions
    first_lat = wind_data[0]['lat']
    n_lons = 0
    for p in wind_data:
        if p['lat'] == first_lat:
            n_lons += 1
        else:
            break

    n_lats = n_points // n_lons
    assert n_lats * n_lons == n_points, f"Grid not rectangular: {n_lats} x {n_lons} != {n_points}"
    log(f"Grid dimensions: {n_lats} x {n_lons}")

    speed_grid = speeds.reshape(n_lats, n_lons)

    grid_lats_asc = np.array([wind_data[i * n_lons]['lat'] for i in range(n_lats)])
    grid_lons_raw = np.array([wind_data[j]['lon'] for j in range(n_lons)])

    log(f"Lats: {grid_lats_asc[0]} to {grid_lats_asc[-1]}")
    log(f"Lons raw: {grid_lons_raw[0]} to {grid_lons_raw[-1]}")

    # Rearrange lons from 0..180,-179..-0.25 to -180..180
    neg_start = None
    for i in range(1, len(grid_lons_raw)):
        if grid_lons_raw[i] < grid_lons_raw[i - 1]:
            neg_start = i
            break

    if neg_start is not None:
        log(f"Lon wraparound at index {neg_start}")
        grid_lons = np.concatenate([grid_lons_raw[neg_start:], grid_lons_raw[:neg_start]])
        speed_grid = np.concatenate([speed_grid[:, neg_start:], speed_grid[:, :neg_start]], axis=1)
    else:
        grid_lons = grid_lons_raw

    # Flip lats to descending (90 to -90)
    grid_lats = grid_lats_asc[::-1]
    speed_grid = speed_grid[::-1, :]

    log(f"Final grid: {speed_grid.shape}")
    log(f"Lats: {grid_lats[0]:.4f} to {grid_lats[-1]:.4f}")
    log(f"Lons: {grid_lons[0]:.4f} to {grid_lons[-1]:.4f}")
    log(f"Speed range: {speed_grid.min():.2f} - {speed_grid.max():.2f} m/s")

    # Extend grid for antimeridian wrapping
    wrap_cols = 60
    speed_grid = np.concatenate([
        speed_grid[:, -wrap_cols:],
        speed_grid,
        speed_grid[:, :wrap_cols]
    ], axis=1)
    grid_lons = np.concatenate([
        grid_lons[-wrap_cols:] - 360,
        grid_lons,
        grid_lons[:wrap_cols] + 360
    ])
    log(f"Extended grid for wrapping: {speed_grid.shape}")

    del wind_data, speeds

    lut = create_vtk_lut()
    tile_renderer = TileRenderer(TILE_SIZE, RENDER_SCALE, PADDING, lut)
    log("VTK image renderer initialized")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TERRAIN_CACHE, exist_ok=True)

    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n_tiles = 2 ** z
        for tx in range(n_tiles):
            os.makedirs(os.path.join(OUTPUT_DIR, str(z), str(tx)), exist_ok=True)
    log("Created directories")

    total_tiles = sum(4 ** z for z in range(MIN_ZOOM, MAX_ZOOM + 1))
    tile_count = 0

    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n_tiles = 2 ** z
        log(f"\nZoom {z}: {n_tiles}x{n_tiles} tiles")

        for tx in range(n_tiles):
            for ty in range(n_tiles):
                tile_count += 1
                tile_dir = os.path.join(OUTPUT_DIR, str(z), str(tx))
                tile_path = os.path.join(tile_dir, f"{ty}.png")
                os.makedirs(tile_dir, exist_ok=True)

                wind_rgb = tile_renderer.render_wind_color(
                    speed_grid, grid_lats, grid_lons, z, tx, ty
                )

                if wind_rgb is None:
                    empty = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                    empty.save(tile_path)
                    continue

                terrain_gray = fetch_terrain_tile(z, tx, ty)
                result = tile_renderer.composite_tile(wind_rgb, terrain_gray)

                img = Image.fromarray(result)
                if img.size != (TILE_SIZE, TILE_SIZE):
                    img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                img.save(tile_path, optimize=True)

                if tile_count % 10 == 0:
                    log(f"  Progress: {tile_count}/{total_tiles} tiles")

    log(f"\n✓ Done: {tile_count} tiles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()