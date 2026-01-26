#!/usr/bin/env python3
"""
Generate wind speed tiles using VTK.

Output: tiles/{z}/{x}/{y}.png (standard XYZ structure)
These tiles are transparent PNGs meant to overlay on a web basemap.
"""

import json
import numpy as np
from pathlib import Path
import vtk  # type: ignore
from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
from PIL import Image  # type: ignore
import argparse


class WindTileGenerator:
    """Generate map tiles from wind data using VTK for smooth interpolation."""
    
    TILE_SIZE = 256
    SPEED_MIN = 0
    SPEED_MAX = 30
    
    def __init__(self, wind_data_path: str):
        self.load_data(wind_data_path)
        self.build_mesh()
    
    def load_data(self, path: str):
        """Load wind data from JSON."""
        print(f"Loading {path}...")
        with open(path) as f:
            data = json.load(f)
        
        self.lats = np.array([p['lat'] for p in data])
        self.lons = np.array([p['lon'] for p in data])
        self.speeds = np.array([p['speed'] for p in data])
        
        print(f"  {len(self.lats)} points")
        print(f"  Speed range: {self.speeds.min():.1f} - {self.speeds.max():.1f} m/s")
    
    def build_mesh(self):
        """Create triangulated mesh from wind points."""
        print("Building mesh...")
        n = len(self.lats)
        
        # Points
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(n)
        for i in range(n):
            points.SetPoint(i, self.lons[i], self.lats[i], 0.0)
        
        # Scalars
        speed_arr = vtk.vtkFloatArray()
        speed_arr.SetName("speed")
        speed_arr.SetNumberOfTuples(n)
        for i, s in enumerate(self.speeds):
            speed_arr.SetValue(i, s)
        
        # PolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().SetScalars(speed_arr)
        
        # Triangulate
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.SetTolerance(0.0001)
        delaunay.Update()
        
        self.mesh = delaunay.GetOutput()
        print(f"  {self.mesh.GetNumberOfCells()} triangles")
    
    def create_lut(self):
        """Create Turbo colormap lookup table."""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        # Turbo colormap
        turbo = [
            (0.190, 0.072, 0.231), (0.232, 0.130, 0.393), (0.266, 0.210, 0.563),
            (0.282, 0.317, 0.715), (0.278, 0.433, 0.836), (0.254, 0.560, 0.912),
            (0.212, 0.684, 0.937), (0.165, 0.793, 0.915), (0.143, 0.877, 0.850),
            (0.194, 0.929, 0.746), (0.326, 0.951, 0.615), (0.501, 0.944, 0.469),
            (0.679, 0.913, 0.329), (0.833, 0.859, 0.210), (0.950, 0.782, 0.133),
            (1.000, 0.682, 0.109), (0.977, 0.561, 0.115), (0.906, 0.431, 0.122),
            (0.808, 0.301, 0.111), (0.692, 0.184, 0.083), (0.566, 0.089, 0.046),
            (0.429, 0.025, 0.013), (0.357, 0.007, 0.004),
        ]
        
        for i in range(256):
            t = i / 255.0
            idx = min(int(t * (len(turbo) - 1)), len(turbo) - 2)
            frac = t * (len(turbo) - 1) - idx
            
            r = turbo[idx][0] + frac * (turbo[idx + 1][0] - turbo[idx][0])
            g = turbo[idx][1] + frac * (turbo[idx + 1][1] - turbo[idx][1])
            b = turbo[idx][2] + frac * (turbo[idx + 1][2] - turbo[idx][2])
            
            lut.SetTableValue(i, r, g, b, 1.0)
        
        lut.SetRange(self.SPEED_MIN, self.SPEED_MAX)
        lut.Build()
        return lut
    
    def render_tile(self, zoom: int, tx: int, ty: int) -> np.ndarray:
        """Render a single tile. Returns RGBA numpy array."""
        # Calculate tile bounds (Web Mercator / equirectangular approximation)
        num_tiles = 2 ** zoom
        
        tile_lon_size = 360.0 / num_tiles
        tile_lat_size = 180.0 / num_tiles
        
        lon_min = -180 + tx * tile_lon_size
        lon_max = lon_min + tile_lon_size
        lat_max = 90 - ty * tile_lat_size  # Y increases downward in tile coords
        lat_min = lat_max - tile_lat_size
        
        # Mapper with smooth interpolation
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        mapper.SetScalarRange(self.SPEED_MIN, self.SPEED_MAX)
        mapper.SetLookupTable(self.create_lut())
        mapper.SetInterpolateScalarsBeforeMapping(True)
        
        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToGouraud()
        actor.GetProperty().LightingOff()
        
        # Renderer with transparent background
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0, 0, 0)
        
        # Orthographic camera for this tile's bounds
        camera = renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        
        camera.SetPosition(center_lon, center_lat, 1)
        camera.SetFocalPoint(center_lon, center_lat, 0)
        camera.SetViewUp(0, 1, 0)
        camera.SetParallelScale((lat_max - lat_min) / 2)
        
        # Render window (offscreen)
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(self.TILE_SIZE, self.TILE_SIZE)
        render_window.SetAlphaBitPlanes(1)
        render_window.Render()
        
        # Capture to image
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(render_window)
        w2i.SetInputBufferTypeToRGBA()
        w2i.Update()
        
        # Convert to numpy
        vtk_image = w2i.GetOutput()
        dims = vtk_image.GetDimensions()
        
        vtk_array = vtk_image.GetPointData().GetScalars()
        rgba = vtk_to_numpy(vtk_array).reshape(dims[1], dims[0], 4)
        
        # Flip vertically (VTK origin is bottom-left)
        rgba = np.flipud(rgba)
        
        # Set black pixels to transparent (where there's no data)
        black_mask = (rgba[:, :, 0] == 0) & (rgba[:, :, 1] == 0) & (rgba[:, :, 2] == 0)
        rgba[black_mask, 3] = 0
        
        render_window.Finalize()
        
        return rgba
    
    def generate_zoom_level(self, zoom: int, output_dir: Path) -> int:
        """Generate all tiles for a zoom level. Returns tile count."""
        num_tiles_x = 2 ** zoom
        num_tiles_y = 2 ** zoom
        
        zoom_dir = output_dir / str(zoom)
        count = 0
        
        for tx in range(num_tiles_x):
            x_dir = zoom_dir / str(tx)
            x_dir.mkdir(parents=True, exist_ok=True)
            
            for ty in range(num_tiles_y):
                rgba = self.render_tile(zoom, tx, ty)
                img = Image.fromarray(rgba.astype(np.uint8), mode='RGBA')
                img.save(x_dir / f"{ty}.png", optimize=True)
                count += 1
        
        return count
    
    def generate_all(self, output_dir: Path, min_zoom: int = 0, max_zoom: int = 4):
        """Generate tiles for all zoom levels."""
        output_dir.mkdir(exist_ok=True)
        
        total = 0
        for z in range(min_zoom, max_zoom + 1):
            print(f"Zoom {z}...", end=" ", flush=True)
            count = self.generate_zoom_level(z, output_dir)
            print(f"{count} tiles")
            total += count
        
        print(f"\n✓ Generated {total} tiles in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate wind speed tiles")
    parser.add_argument("input", help="Path to wind_data.json")
    parser.add_argument("-o", "--output", default="tiles", help="Output directory")
    parser.add_argument("--min-zoom", type=int, default=0)
    parser.add_argument("--max-zoom", type=int, default=4)
    
    args = parser.parse_args()
    
    generator = WindTileGenerator(args.input)
    generator.generate_all(
        Path(args.output),
        min_zoom=args.min_zoom,
        max_zoom=args.max_zoom
    )


if __name__ == "__main__":
    main()