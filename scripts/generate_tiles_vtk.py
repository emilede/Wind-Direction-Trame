#!/usr/bin/env python3
"""
Generate tile pyramid for wind visualization using VTK.
Composites wind color with grayscale terrain for Zoom Earth-like effect.
"""

import json
import sys
import os
import numpy as np
import mercantile
import vtk
from PIL import Image
from scipy.ndimage import gaussian_filter
import urllib.request

# ============ CONFIG ============
TILE_SIZE = 256       # Output tile size
RENDER_SCALE = 4      # 4x oversampling - Lanczos downsample smooths edges
PADDING = 32          # Padding for blur
MIN_ZOOM = 2
MAX_ZOOM = 3          # 80 tiles (reduced for faster rendering)
OUTPUT_DIR = os.path.abspath('data/tiles')
TERRAIN_CACHE = os.path.abspath('data/terrain_cache')

# Wind color settings
GLOW_BLUR = 1      # Minimal blur - preserve sharp transitions
DETAIL_BLUR = 0.3  # Light smoothing
SPEED_GAMMA = 0.7
# ================================

def log(msg):
    print(msg)
    sys.stdout.flush()


def fetch_terrain_tile(z, x, y):
    """Fetch grayscale hillshade tile from ESRI."""
    cache_path = os.path.join(TERRAIN_CACHE, str(z), str(x), f"{y}.png")
    
    if os.path.exists(cache_path):
        img = Image.open(cache_path).convert('L')  # Grayscale
        # Resize to match our tile size if needed
        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
        return np.array(img)
    
    # ESRI World Hillshade DARK (better for color overlay)
    url = f"https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade_Dark/MapServer/tile/{z}/{y}/{x}"
    
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        img = Image.open(cache_path).convert('L')
        # Resize to match our tile size if needed
        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        log(f"  Warning: terrain fetch failed for {z}/{x}/{y}: {e}")
        # Return neutral gray if fetch fails
        return np.full((TILE_SIZE, TILE_SIZE), 128, dtype=np.uint8)


def create_vtk_lut():
    """Create VTK lookup table with bell curve saturation.
    Darkest at 0 and 35, brightest around 15-20 m/s (middle)."""
    colors = [
        # 0 m/s - bluish purple (DARK)
        (0.000, (30, 20, 80)),
        
        # 1 m/s - dark blue
        (0.029, (20, 30, 100)),
        
        # 3 m/s - blue
        (0.086, (35, 60, 140)),
        
        # 5 m/s - light blue
        (0.143, (50, 100, 180)),
        
        # 7.5 m/s - light blue (brighter)
        (0.214, (60, 130, 190)),
        
        # 10 m/s - cyan (bright)
        (0.286, (70, 190, 210)),
        
        # 11 m/s - cyan-green (bright)
        (0.314, (60, 210, 160)),
        
        # 12 m/s - green (toned down)
        (0.343, (50, 180, 70)),
        
        # 13 m/s - bright green (toned down)
        (0.371, (100, 190, 55)),
        
        # 14 m/s - lime (toned down)
        (0.400, (150, 200, 50)),
        
        # 15 m/s - yellow (BRIGHTEST)
        (0.429, (240, 240, 70)),
        
        # 16 m/s - yellow-gold (BRIGHTEST)
        (0.457, (250, 220, 60)),
        
        # 17.5 m/s - gold (BRIGHTEST - peak)
        (0.500, (255, 200, 50)),
        
        # 19 m/s - gold-orange (bright)
        (0.543, (250, 160, 45)),
        
        # 20 m/s - orange (bright)
        (0.571, (245, 120, 45)),
        
        # 21.5 m/s - orange-red (getting darker)
        (0.614, (220, 90, 50)),
        
        # 23 m/s - red-orange (darker)
        (0.657, (190, 65, 55)),
        
        # 25 m/s - deep red (dark)
        (0.714, (160, 50, 60)),
        
        # 27.5 m/s - red-magenta (darker)
        (0.786, (140, 40, 75)),
        
        # 30 m/s - magenta (dark)
        (0.857, (120, 35, 85)),
        
        # 32.5 m/s - dark magenta (DARK)
        (0.929, (100, 30, 80)),
        
        # 35 m/s - deep magenta (DARKEST)
        (1.000, (80, 25, 70)),
    ]
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(1024)  # Higher resolution for smoother gradients
    lut.SetRange(0, 35)  # 0-35 m/s
    
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
                
                lut.SetTableValue(i, r, g, b, 1.0)  # Fully opaque
                break
    
    lut.Build()
    return lut


class TileRenderer:
    """Reusable VTK renderer for generating tiles."""
    
    def __init__(self, tile_size, render_scale, padding, lut):
        self.tile_size = tile_size
        self.padding = padding
        self.render_scale = render_scale
        self.lut = lut
        
        self.padded_size = tile_size + 2 * padding
        self.render_size = self.padded_size * render_scale
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(self.render_size, self.render_size)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.render_window.AddRenderer(self.renderer)
        
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(self.render_size/2, self.render_size/2, 100)
        camera.SetFocalPoint(self.render_size/2, self.render_size/2, 0)
        camera.SetParallelScale(self.render_size/2)
        
        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(self.render_window)
        self.w2i.SetInputBufferTypeToRGB()  # No alpha needed
    
    def render_wind_color(self, lons, lats, speeds, z, x, y):
        """Render wind as solid color (no alpha)."""
        bounds = mercantile.bounds(x, y, z)
        lon_min, lat_min, lon_max, lat_max = bounds.west, bounds.south, bounds.east, bounds.north
        
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        pad_frac = self.padding / self.tile_size
        geo_lon_pad = lon_range * (pad_frac + 0.1)
        geo_lat_pad = lat_range * (pad_frac + 0.1)
        
        mask = (
            (lons >= lon_min - geo_lon_pad) & (lons <= lon_max + geo_lon_pad) &
            (lats >= lat_min - geo_lat_pad) & (lats <= lat_max + geo_lat_pad)
        )
        
        if mask.sum() < 4:
            return None
        
        tile_lons = lons[mask]
        tile_lats = lats[mask]
        tile_speeds = speeds[mask]
        
        n = 2 ** z
        
        # Calculate padded bounds (in geographic coordinates)
        pad_frac = self.padding / self.tile_size
        padded_lon_min = lon_min - lon_range * pad_frac
        padded_lon_max = lon_max + lon_range * pad_frac
        
        def lat_to_merc_y(lat):
            lat_rad = np.radians(np.clip(lat, -85.05, 85.05))
            return (1 - np.log(np.tan(lat_rad) + 1/np.cos(lat_rad)) / np.pi) / 2
        
        def merc_y_to_lat(merc_y):
            return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * merc_y))))
        
        tile_merc_min = y / n
        tile_merc_max = (y + 1) / n
        merc_range = tile_merc_max - tile_merc_min
        padded_merc_min = tile_merc_min - merc_range * pad_frac
        padded_merc_max = tile_merc_max + merc_range * pad_frac
        
        # Project source points to pixel coordinates
        px = (tile_lons - padded_lon_min) / (padded_lon_max - padded_lon_min) * self.render_size
        merc_y = lat_to_merc_y(tile_lats)
        py = (merc_y - padded_merc_min) / (padded_merc_max - padded_merc_min) * self.render_size
        
        # Create source points for Delaunay triangulation
        points = vtk.vtkPoints()
        for i in range(len(px)):
            points.InsertNextPoint(px[i], self.render_size - py[i], 0)
        
        scalars = vtk.vtkFloatArray()
        scalars.SetName("speed")
        for s in tile_speeds:
            scalars.InsertNextValue(s)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(scalars)
        
        # Triangulate the scattered points
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.SetTolerance(0.001)
        delaunay.Update()
        
        # Subdivide triangles for smoother color transitions
        subdivide = vtk.vtkLinearSubdivisionFilter()
        subdivide.SetInputConnection(delaunay.GetOutputPort())
        subdivide.SetNumberOfSubdivisions(2)  # Each triangle -> 16 smaller triangles
        subdivide.Update()
        
        # DIAGNOSTIC
        if not hasattr(self, '_diag_count'):
            self._diag_count = 0
        self._diag_count += 1
        
        if self._diag_count <= 3 or tile_speeds.max() > 20:
            log(f"\n  === DIAGNOSTIC for tile {z}/{x}/{y} ===")
            log(f"  Input speeds: min={tile_speeds.min():.2f}, max={tile_speeds.max():.2f}")
            d_scalars = subdivide.GetOutput().GetPointData().GetScalars()
            if d_scalars:
                d_range = d_scalars.GetRange()
                log(f"  Subdivided scalars: min={d_range[0]:.2f}, max={d_range[1]:.2f}")
            log(f"  ===========================")
        
        # Render subdivided mesh with Gouraud shading
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(subdivide.GetOutputPort())
        mapper.SetLookupTable(self.lut)
        mapper.SetScalarRange(0, 35)
        mapper.SetInterpolateScalarsBeforeMapping(True)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToGouraud()
        
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.render_window.Render()
        
        self.w2i.Modified()
        self.w2i.Update()
        
        vtk_image = self.w2i.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        arr = np.frombuffer(vtk_array, dtype=np.uint8).reshape(height, width, 3)
        arr = np.flipud(arr.copy())
        
        # Return padded image - blur and crop happens in composite_tile
        return arr
    
    def composite_tile(self, wind_rgb_padded, terrain_gray):
        """Blend wind color with terrain - apply blur BEFORE cropping."""
        
        # Apply glow blur to padded image (before crop!)
        glow = np.zeros_like(wind_rgb_padded, dtype=float)
        for c in range(3):
            glow[:, :, c] = gaussian_filter(wind_rgb_padded[:, :, c].astype(float), sigma=GLOW_BLUR)
        
        # Apply detail blur to padded image
        detail = np.zeros_like(wind_rgb_padded, dtype=float)
        for c in range(3):
            detail[:, :, c] = gaussian_filter(wind_rgb_padded[:, :, c].astype(float), sigma=DETAIL_BLUR)
        
        # Combine glow and detail - favor detail to preserve reds
        wind_combined = (glow * 0.2 + detail * 0.8)
        
        # NOW crop the padding (after blur)
        crop_start = self.padding * self.render_scale
        crop_end = crop_start + self.tile_size * self.render_scale
        wind_cropped = wind_combined[crop_start:crop_end, crop_start:crop_end]
        
        # Resize terrain to match cropped wind
        if terrain_gray.shape[0] != wind_cropped.shape[0]:
            terrain_img = Image.fromarray(terrain_gray)
            terrain_img = terrain_img.resize((wind_cropped.shape[1], wind_cropped.shape[0]), Image.LANCZOS)
            terrain_gray = np.array(terrain_img)
        
        # Normalize terrain to 0-1, boost it significantly
        terrain_norm = terrain_gray.astype(float) / 255.0
        terrain_norm = np.clip(terrain_norm * 2.0, 0, 1)  # Brighten terrain a lot
        
        # Screen blend: result = 1 - (1-terrain) * (1-wind)
        result = np.zeros((wind_cropped.shape[0], wind_cropped.shape[1], 3), dtype=float)
        for c in range(3):
            w = wind_cropped[:, :, c] / 255.0
            t = terrain_norm
            result[:, :, c] = 1 - (1 - t * 0.3) * (1 - w)
        
        # Mild brightness boost - preserve dark colors
        result = np.clip(result * 255 * 1.1, 0, 255).astype(np.uint8)
        
        return result


def main():
    log("Loading wind data...")
    with open('data/wind_data.json') as f:
        wind_data = json.load(f)
    
    lons = np.array([p['lon'] for p in wind_data])
    lats = np.array([p['lat'] for p in wind_data])
    speeds = np.array([p['speed'] for p in wind_data])
    log(f"Loaded {len(wind_data):,} points")
    log(f"GLOBAL SPEEDS: min={speeds.min():.2f}, max={speeds.max():.2f}, mean={speeds.mean():.2f} m/s")
    
    # Handle antimeridian wrapping
    east_mask = lons > 170
    west_mask = lons < -170
    
    all_lons = lons.copy()
    all_lats = lats.copy()
    all_speeds = speeds.copy()
    
    if east_mask.any():
        all_lons = np.concatenate([all_lons, lons[east_mask] - 360])
        all_lats = np.concatenate([all_lats, lats[east_mask]])
        all_speeds = np.concatenate([all_speeds, speeds[east_mask]])
    if west_mask.any():
        all_lons = np.concatenate([all_lons, lons[west_mask] + 360])
        all_lats = np.concatenate([all_lats, lats[west_mask]])
        all_speeds = np.concatenate([all_speeds, speeds[west_mask]])
    
    lons, lats, speeds = all_lons, all_lats, all_speeds
    log(f"After wrapping: {len(lons):,} points")
    
    # Create renderer
    lut = create_vtk_lut()
    tile_renderer = TileRenderer(TILE_SIZE, RENDER_SCALE, PADDING, lut)
    log("VTK renderer initialized")
    
    # Create directories
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
                
                # Ensure directory exists
                os.makedirs(tile_dir, exist_ok=True)
                
                # Render wind color
                wind_rgb = tile_renderer.render_wind_color(lons, lats, speeds, z, tx, ty)
                
                if wind_rgb is None:
                    empty = Image.new('RGB', (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                    empty.save(tile_path)
                    continue
                
                # Fetch terrain
                terrain_gray = fetch_terrain_tile(z, tx, ty)
                
                # Composite
                result = tile_renderer.composite_tile(wind_rgb, terrain_gray)
                
                # Save
                img = Image.fromarray(result)
                if img.size != (TILE_SIZE, TILE_SIZE):
                    img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)
                img.save(tile_path, optimize=True)
                
                if tile_count % 10 == 0:
                    log(f"  Progress: {tile_count}/{total_tiles} tiles")
    
    log(f"\n✓ Done: {tile_count} tiles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()