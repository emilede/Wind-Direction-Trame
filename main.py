"""
Wind visualization with pre-rendered tiles.
Tooltip shows wind speed/direction on hover using Trame state.

TOOLTIP ARCHITECTURE (for thesis documentation):
=================================================
The tooltip uses a hybrid client-server approach within the Trame framework:

CLIENT SIDE (Vue mousemove handler on VContainer):
1. VContainer wraps the Leaflet map and captures mousemove events
2. On hover, we find a visible Leaflet tile image in the DOM
3. Parse z/x/y from the tile's URL (e.g., /tiles/3/4/2.png)
4. Calculate the mouse's fractional position within that tile
5. Convert tile coords + fraction to lat/lng using Web Mercator math
6. Send [lat, lng, clientX, clientY] to server via trame state (mouse_data)

SERVER SIDE (Python @state.change handler):
1. Receives lat/lng from client
2. Looks up nearest wind data point in a pre-built 0.5° grid
3. Computes speed (mph), compass direction, and arrow rotation
4. Updates tooltip state variables (text, position, visibility)

CLIENT SIDE (Vue reactive rendering):
1. Tooltip div is bound to state variables via v-show, v-text, and dynamic style
2. Positioned at cursor via fixed positioning with state-driven left/top

Key insight: By reading tile coordinates directly from the DOM (not from
trame state's `center` which can be stale after dragging), the lat/lng
calculation is always accurate regardless of map interaction state.
=================================================
"""

import os
import json
import time
import math
import numpy as np
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, leaflet3 as leaflet, html as html_widgets

# Cache buster
CACHE_BUST = int(time.time())

server = get_server(client_type="vue3")
state = server.state
state.trame__title = "Wind Speed"

# ============ WIND DATA LOOKUP ============

print("Loading wind data for hover lookups...")
with open('data/wind_data.json') as f:
    wind_data = json.load(f)

# Build lookup grid at 0.5° resolution
LOOKUP_STEP = 0.5
lookup_grid = {}
for p in wind_data:
    lat_key = round(round(p['lat'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    lon_key = round(round(p['lon'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    key = (lat_key, lon_key)
    if key not in lookup_grid:
        lookup_grid[key] = (round(p['speed'], 1), round(p['direction']))

print(f"Lookup grid: {len(lookup_grid)} points")
del wind_data  # Free memory

COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def lookup_wind(lat, lon):
    """Look up wind speed and direction at nearest grid point."""
    lat_key = round(round(lat / LOOKUP_STEP) * LOOKUP_STEP, 1)

    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    lon_key = round(round(lon / LOOKUP_STEP) * LOOKUP_STEP, 1)

    return lookup_grid.get((lat_key, lon_key))


# ============ TRAME STATE ============

state.tooltip_visible = False
state.tooltip_text = ""
state.tooltip_dir = ""
state.tooltip_arrow_rot = 0
state.tooltip_x = 0
state.tooltip_y = 0
state.mouse_data = None
state.debug_msg = "Move mouse over map..."


@state.change("mouse_data")
def on_mouse_move(mouse_data, **kwargs):
    if mouse_data is None:
        state.tooltip_visible = False
        return

    try:
        lat, lon, cx, cy = mouse_data
        result = lookup_wind(lat, lon)

        if result:
            speed, direction = result
            compass = COMPASS[round(direction / 22.5) % 16]
            state.tooltip_visible = True
            state.tooltip_text = f"{round(speed * 2.237)} mph"
            state.tooltip_dir = compass
            state.tooltip_arrow_rot = (direction + 180) % 360
            state.tooltip_x = int(cx)
            state.tooltip_y = int(cy)
        else:
            state.tooltip_visible = False
    except Exception:
        state.tooltip_visible = False


# ============ TILE SERVING ============

async def serve_tile(request):
    z = request.match_info['z']
    x = request.match_info['x']
    y = request.match_info['y']
    tile_path = f"./data/tiles/{z}/{x}/{y}.png"

    if os.path.exists(tile_path):
        response = web.FileResponse(tile_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return web.Response(status=404)


async def serve_border_tile(request):
    z = request.match_info['z']
    x = request.match_info['x']
    y = request.match_info['y']
    tile_path = f"./data/border_tiles/{z}/{x}/{y}.png"

    if os.path.exists(tile_path):
        response = web.FileResponse(tile_path)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    return web.Response(status=404)


@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)
    wslink_server.app.router.add_get("/borders/{z}/{x}/{y}.png", serve_border_tile)


# ============ UI LAYOUT ============

# JavaScript for coordinate calculation from tile positions
# This avoids needing the Leaflet map instance entirely
MOUSEMOVE_JS = """
    if ($event.buttons > 0) {
        tooltip_visible = false;
        return;
    }
    if (!window._ttThrottle || Date.now() - window._ttThrottle > 60) {
        window._ttThrottle = Date.now();

        var tiles = $event.currentTarget.querySelectorAll('.leaflet-tile');
        var found = false;

        for (var i = 0; i < tiles.length; i++) {
            var tile = tiles[i];
            var src = tile.src || tile.getAttribute('src') || '';

            var match = src.match(/\\/tiles\\/(\\d+)\\/(\\d+)\\/(\\d+)\\.png/);
            if (!match) continue;

            var tz = parseInt(match[1]);
            var tx = parseInt(match[2]);
            var ty = parseInt(match[3]);

            var tileRect = tile.getBoundingClientRect();
            if (tileRect.width < 1 || tileRect.height < 1) continue;

            var fracX = ($event.clientX - tileRect.left) / tileRect.width;
            var fracY = ($event.clientY - tileRect.top) / tileRect.height;

            if (fracX < -0.5 || fracX > 1.5 || fracY < -0.5 || fracY > 1.5) continue;

            var n = Math.pow(2, tz);
            var lon = (tx + fracX) / n * 360 - 180;
            var latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * (ty + fracY) / n)));
            var lat = latRad * 180 / Math.PI;

            mouse_data = [lat, lon, $event.clientX, $event.clientY];
            debug_msg = 'lat: ' + lat.toFixed(2) + ' lon: ' + lon.toFixed(2);
            found = true;
            break;
        }

        if (!found) {
            debug_msg = 'No tile under cursor (found ' + tiles.length + ' tiles)';
        }
    }
"""

with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Speed")

    with layout.content:
        with v3.VContainer(
            fluid=True,
            classes="fill-height pa-0",
            style="position: relative;",
            mousemove=MOUSEMOVE_JS,
            mouseleave="mouse_data = null; tooltip_visible = false",
        ):
            with leaflet.LMap(
                zoom=("zoom", 3),
                center=("center", [20, 0]),
                world_copy_jump=True,
                max_zoom=3,
                min_zoom=3,
                style="height: 100%; width: 100%;",
            ):
                leaflet.LTileLayer(
                    url=("wind_url", f"/tiles/{{z}}/{{x}}/{{y}}.png?v={CACHE_BUST}"),
                    opacity=1.0,
                    minZoom=3,
                    maxZoom=5,
                )
                leaflet.LTileLayer(
                    url=("border_url", f"/borders/{{z}}/{{x}}/{{y}}.png?v={CACHE_BUST}"),
                    opacity=1.0,
                    minZoom=3,
                    maxZoom=5,
                )

        # Tooltip
        html_widgets.Div(
            v_show=("tooltip_visible",),
            v_text=("tooltip_text + ' ' + tooltip_dir",),
            style=("`position:fixed; left:${tooltip_x}px; top:${tooltip_y - 45}px; transform:translateX(-50%); background:rgba(30,30,30,0.88); color:white; padding:5px 14px; border-radius:18px; font-size:13px; font-weight:500; pointer-events:none; z-index:10000; white-space:nowrap; backdrop-filter:blur(4px); font-family:-apple-system,BlinkMacSystemFont,sans-serif;`",),
        )

        # Wind speed legend (bottom-left)
        with v3.VSheet(
            color="transparent",
            style="position:fixed; bottom:30px; left:10px; z-index:10000;",
        ):
            with v3.VCard(
                color="rgba(20,15,30,0.85)",
                rounded="lg",
                elevation=0,
                style="padding:8px 12px 6px 12px; backdrop-filter:blur(4px);",
            ):
                with v3.VRow(no_gutters=True, align="center", classes="mb-1"):
                    v3.VCardText(
                        "mph",
                        style="color:white; font-size:11px; font-weight:600; padding:0; min-width:30px;",
                    )
                    html_widgets.Div(
                        style=(
                            "width:280px; height:14px; border-radius:3px;"
                            "background: linear-gradient(to right,"
                            "rgb(15,35,55) 0%,"
                            "rgb(25,80,105) 8%,"
                            "rgb(35,110,135) 13%,"
                            "rgb(40,145,130) 18%,"
                            "rgb(60,180,65) 25%,"
                            "rgb(170,210,45) 35%,"
                            "rgb(245,200,45) 43%,"
                            "rgb(245,160,40) 50%,"
                            "rgb(235,115,40) 56%,"
                            "rgb(215,80,40) 63%,"
                            "rgb(155,40,55) 73%,"
                            "rgb(95,25,60) 86%,"
                            "rgb(50,15,40) 100%);"
                        ),
                    )
                with v3.VRow(no_gutters=True, style="margin-left:30px; width:280px;", justify="space-between"):
                    for val in ["0", "10", "20", "30", "40", "50", "60", "70", "80"]:
                        v3.VCardText(
                            val,
                            style="color:white; font-size:10px; padding:0; text-align:center; min-width:0;",
                        )

if __name__ == "__main__":
    server.start()