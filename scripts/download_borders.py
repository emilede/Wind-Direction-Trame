#!/usr/bin/env python3
"""
Download Natural Earth country boundary lines GeoJSON.
"""

import urllib.request
import os

# Create directory
os.makedirs('data/geojson', exist_ok=True)

# Natural Earth 50m boundary lines (better for zoom 5)
url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_boundary_lines_land.geojson"
output = "data/geojson/ne_50m_admin_0_boundary_lines_land.geojson"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, output)
print(f"Saved to {output}")

# Check file size
size = os.path.getsize(output)
print(f"File size: {size / 1024:.1f} KB")

# Also get coastlines
url2 = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson"
output2 = "data/geojson/ne_50m_coastline.geojson"

print(f"\nDownloading {url2}...")
urllib.request.urlretrieve(url2, output2)
print(f"Saved to {output2}")

size2 = os.path.getsize(output2)
print(f"File size: {size2 / 1024:.1f} KB")

print("\n✓ Done! Borders ready for use.")