#!/usr/bin/env python3
"""
Generate border tiles - transparent background with black country/coastline borders.
"""

import json
import os
import sys
import mercantile
from PIL import Image, ImageDraw
import math

# ============ CONFIG ============
TILE_SIZE = 256       # Match wind tiles
MIN_ZOOM = 2
MAX_ZOOM = 3
OUTPUT_DIR = 'data/border_tiles'
GEOJSON_DIR = 'data/geojson'

# Line style
LINE_COLOR = (0, 0, 0, 200)  # Black with some transparency
LINE_WIDTH = 1
# ================================

def log(msg):
    print(msg)
    sys.stdout.flush()


def lon_to_x(lon, zoom):
    """Convert longitude to tile x coordinate."""
    return (lon + 180.0) / 360.0 * (2 ** zoom)


def lat_to_y(lat, zoom):
    """Convert latitude to tile y coordinate."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n


def project_to_tile(lon, lat, tx, ty, zoom, tile_size):
    """Project lon/lat to pixel coordinates within a tile."""
    x = lon_to_x(lon, zoom)
    y = lat_to_y(lat, zoom)
    
    # Convert to pixel within tile
    px = (x - tx) * tile_size
    py = (y - ty) * tile_size
    
    return px, py


def load_geojson(filename):
    """Load GeoJSON file."""
    path = os.path.join(GEOJSON_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_lines_from_geojson(geojson):
    """Extract all line coordinates from GeoJSON."""
    lines = []
    
    if not geojson or 'features' not in geojson:
        return lines
    
    for feature in geojson['features']:
        geom = feature.get('geometry', {})
        geom_type = geom.get('type', '')
        coords = geom.get('coordinates', [])
        
        if geom_type == 'LineString':
            lines.append(coords)
        elif geom_type == 'MultiLineString':
            lines.extend(coords)
        elif geom_type == 'Polygon':
            # Exterior ring
            if coords:
                lines.append(coords[0])
        elif geom_type == 'MultiPolygon':
            for polygon in coords:
                if polygon:
                    lines.append(polygon[0])
    
    return lines


def render_tile(lines, tx, ty, zoom):
    """Render border lines for a single tile."""
    # Create transparent image
    img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Get tile bounds
    bounds = mercantile.bounds(tx, ty, zoom)
    
    # Expand bounds slightly for lines that cross tile edges
    margin = 0.5  # degrees
    min_lon = bounds.west - margin
    max_lon = bounds.east + margin
    min_lat = bounds.south - margin
    max_lat = bounds.north + margin
    
    for line in lines:
        # Filter to lines that might intersect this tile
        line_lons = [p[0] for p in line]
        line_lats = [p[1] for p in line]
        
        if max(line_lons) < min_lon or min(line_lons) > max_lon:
            continue
        if max(line_lats) < min_lat or min(line_lats) > max_lat:
            continue
        
        # Project points to tile pixels
        pixels = []
        for lon, lat in line:
            px, py = project_to_tile(lon, lat, tx, ty, zoom, TILE_SIZE)
            pixels.append((px, py))
        
        # Draw line segments
        if len(pixels) >= 2:
            draw.line(pixels, fill=LINE_COLOR, width=LINE_WIDTH)
    
    return img


def main():
    log("Loading GeoJSON files...")
    
    # Load borders and coastlines (50m resolution for zoom 5)
    borders = load_geojson('ne_50m_admin_0_boundary_lines_land.geojson')
    coastlines = load_geojson('ne_50m_coastline.geojson')
    
    # Fallback to 110m if 50m not available
    if not borders:
        borders = load_geojson('ne_110m_admin_0_boundary_lines_land.geojson')
    if not coastlines:
        coastlines = load_geojson('ne_110m_coastline.geojson')
    
    if not borders and not coastlines:
        log("❌ No GeoJSON files found! Run download_borders.py first.")
        sys.exit(1)
    
    # Extract all lines
    all_lines = []
    
    if borders:
        border_lines = extract_lines_from_geojson(borders)
        all_lines.extend(border_lines)
        log(f"  Borders: {len(border_lines)} lines")
    
    if coastlines:
        coast_lines = extract_lines_from_geojson(coastlines)
        all_lines.extend(coast_lines)
        log(f"  Coastlines: {len(coast_lines)} lines")
    
    log(f"Total: {len(all_lines)} lines to render")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n_tiles = 2 ** z
        for tx in range(n_tiles):
            os.makedirs(os.path.join(OUTPUT_DIR, str(z), str(tx)), exist_ok=True)
    
    # Generate tiles
    total_tiles = sum(4 ** z for z in range(MIN_ZOOM, MAX_ZOOM + 1))
    tile_count = 0
    
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        n_tiles = 2 ** z
        log(f"\nZoom {z}: {n_tiles}x{n_tiles} tiles")
        
        for tx in range(n_tiles):
            for ty in range(n_tiles):
                tile_count += 1
                tile_path = os.path.join(OUTPUT_DIR, str(z), str(tx), f"{ty}.png")
                
                # Render tile
                img = render_tile(all_lines, tx, ty, z)
                img.save(tile_path, 'PNG')
                
                if tile_count % 20 == 0:
                    log(f"  Progress: {tile_count}/{total_tiles} tiles")
    
    log(f"\n✓ Done: {tile_count} border tiles saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()