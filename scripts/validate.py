#!/usr/bin/env python3
"""
Validate wind_data.json has sufficient data points and correct structure.
"""

import json
import sys

def main():
    filename = 'data/wind_data.json'
    
    print(f"Checking {filename}...")
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)
    
    # Check point count
    count = len(data)
    print(f"Total points: {count:,}")
    
    if count < 10000:
        print(f"❌ Too few points (need at least 10,000 for good coverage)")
        sys.exit(1)
    elif count < 50000:
        print(f"⚠️  Moderate coverage ({count:,} points)")
    else:
        print(f"✓ Good coverage ({count:,} points)")
    
    # Check structure of first point
    if count > 0:
        sample = data[0]
        required_keys = ['lat', 'lon', 'u', 'v', 'speed', 'direction']
        missing = [k for k in required_keys if k not in sample]
        
        if missing:
            print(f"❌ Missing keys: {missing}")
            sys.exit(1)
        else:
            print(f"✓ Structure valid: {list(sample.keys())}")
    
    # Check lat/lon ranges
    lats = [p['lat'] for p in data]
    lons = [p['lon'] for p in data]
    
    print(f"Latitude range: {min(lats):.1f} to {max(lats):.1f}")
    print(f"Longitude range: {min(lons):.1f} to {max(lons):.1f}")
    
    # Check for global coverage
    if max(lats) < 80 or min(lats) > -80:
        print("⚠️  Limited latitude coverage")
    else:
        print("✓ Global latitude coverage")
    
    if max(lons) - min(lons) < 350:
        print("⚠️  Limited longitude coverage")
    else:
        print("✓ Global longitude coverage")
    
    # Show sample points
    print(f"\nSample points:")
    for i in [0, count//2, count-1]:
        p = data[i]
        print(f"  [{i}] lat={p['lat']:.1f}, lon={p['lon']:.1f}, speed={p['speed']:.1f} m/s, dir={p['direction']:.0f}°")
    
    print(f"\n✓ wind_data.json is ready to use")

if __name__ == '__main__':
    main()