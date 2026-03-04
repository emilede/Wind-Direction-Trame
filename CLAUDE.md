# Wind Direction Trame

Interactive global wind visualization built with Trame, Leaflet, and VTK. Thesis project.

Always use Trame and VTK or related tools made to be used with Trame or VTK. If you want
to do something involving another tool, or think that vtk/trame is inhibiting/limiting factor
tell me that, and don't write any code unless I confirm that you can. 

## Quick Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py

# Data pipeline (from GRIB2 source)
python scripts/convert.py              # GRIB2 → JSON + NPZ
python scripts/generate_tiles_vtk.py   # NPZ → tile pyramid (zoom 3-5)
python scripts/generate_border_tiles.py # GeoJSON → border tile overlays
```

## Project Structure

- `main.py` — Main Trame app: tile server, Leaflet map, particle animation, tooltips
- `scripts/` — Data processing pipeline (convert, tile generation, borders)
- `data/tiles/{z}/{x}/{y}.png` — Pre-rendered wind speed tiles (zoom 3-5)
- `data/border_tiles/` — Country boundary overlay tiles
- `data/wind_grid.npz` — Full resolution grid (0.0625°) for tile rendering
- `data/wind_data.json` — Subsampled grid (0.5°) for tooltip lookups
- `www/map.html` — Standalone Leaflet map reference

## Tech Stack

- **Python**: Trame 3, trame-vuetify (Vue 3), trame-leaflet, VTK, NumPy, SciPy, Pillow, aiohttp
- **Frontend**: Vue 3 (via Trame), Leaflet 1.9.4, Canvas API for particle animation
- **Data**: GRIB2 (GFS 10m wind) → xarray/cfgrib → NPZ/JSON → PNG tiles

## Key Conventions

- Wind components: u (east-west), v (north-south); speed in m/s, displayed as mph
- Grid layout: lats descending (90→-90), lons ascending (-180→180)
- Tiles: Web Mercator, 256×256px, rendered at 4x then downsampled
- Color scheme: Zoom Earth palette, 0-35 m/s range, 17-point lookup table
- Particle system: 1500 particles, JavaScript Canvas, bilinear wind field interpolation
- Code sections marked with `# === SECTION ===` headers
