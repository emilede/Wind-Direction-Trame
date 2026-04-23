"""
Wind visualization with pre-rendered tiles and animated wind particles.

Uses trame-native mechanisms:
- server.enable_module() to serve and auto-load the particle JS module
- client.ClientStateChange for reactive state watching
- Trame state for particle control (particle_active, particle_count)
"""

import os
import json
import time
import math
import resource
import numpy as np
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, leaflet3 as leaflet, html as html_widgets
from trame.widgets import client

_app_start = time.perf_counter()
_t_imports = time.perf_counter()
print(f"BENCHMARK: Imports: {(_t_imports - _app_start)*1000:.1f}ms")

CACHE_BUST = int(time.time())

server = get_server(client_type="vue3")
state = server.state
state.trame__title = "Wind Speed"

_t_pipeline = time.perf_counter()
print(f"BENCHMARK: Pipeline setup: {(_t_pipeline - _app_start)*1000:.1f}ms")

# ============ WIND DATA LOOKUP + WIND FIELD ============

LOOKUP_STEP = 0.5
FIELD_STEP = 1.0
LOOKUP_CACHE = 'data/lookup_cache.json'
WIND_FIELD_OUT = 'www/wind_field.json'

os.makedirs('www', exist_ok=True)

if os.path.exists(LOOKUP_CACHE) and os.path.exists(WIND_FIELD_OUT):
    with open(LOOKUP_CACHE) as f:
        raw = json.load(f)
    lookup_grid = {tuple(float(x) for x in k.split(',')): v for k, v in raw.items()}
else:
    with open('data/wind_data.json') as f:
        wind_data = json.load(f)

    lookup_grid = {}
    field_u = {}
    field_v = {}

    for p in wind_data:
        lat_key = round(round(p['lat'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
        lon_key = round(round(p['lon'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
        key = (lat_key, lon_key)
        if key not in lookup_grid:
            lookup_grid[key] = (round(p['speed'], 1), round(p['direction']))

        flat = float(int(round(p['lat'] / FIELD_STEP))) * FIELD_STEP
        flon = float(int(round(p['lon'] / FIELD_STEP))) * FIELD_STEP
        fkey = (flat, flon)
        if fkey not in field_u:
            field_u[fkey] = round(p['u'], 2)
            field_v[fkey] = round(p['v'], 2)

    with open(LOOKUP_CACHE, 'w') as f:
        json.dump({f"{k[0]},{k[1]}": list(v) for k, v in lookup_grid.items()}, f)

    u_arr = []
    v_arr = []
    for lat_i in range(90, -91, -1):
        for lon_i in range(-180, 181, 1):
            key = (float(lat_i), float(lon_i))
            u_arr.append(field_u.get(key, 0.0))
            v_arr.append(field_v.get(key, 0.0))

    wind_field_json = json.dumps({
        'lat_min': -90, 'lat_max': 90, 'lat_step': 1.0,
        'lon_min': -180, 'lon_max': 180, 'lon_step': 1.0,
        'n_lats': 181, 'n_lons': 361,
        'u': u_arr, 'v': v_arr
    })
    with open(WIND_FIELD_OUT, 'w') as f:
        f.write(wind_field_json)

    del wind_data, field_u, field_v, u_arr, v_arr, wind_field_json

_t_data = time.perf_counter()
print(f"BENCHMARK: Data loaded: {(_t_data - _app_start)*1000:.1f}ms")

COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def lookup_wind(lat, lon):
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

# Particle control state (reactive via ClientStateChange)
state.particle_active = True
state.particle_count = 1500


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
        return web.FileResponse(tile_path, headers={'Cache-Control': 'public, max-age=3600'})
    return web.Response(status=404)


async def serve_border_tile(request):
    z = request.match_info['z']
    x = request.match_info['x']
    y = request.match_info['y']
    tile_path = f"./data/border_tiles/{z}/{x}/{y}.png"
    if os.path.exists(tile_path):
        return web.FileResponse(tile_path, headers={'Cache-Control': 'public, max-age=3600'})
    return web.Response(status=404)


async def receive_bench(request):
    """Receive FPS samples from the browser and print stats matching the C1 format."""
    data = await request.json()
    samples = data.get('samples', [])
    arr = np.array(samples)
    print(f"\n[BENCH] FPS over {len(arr)} samples ({len(arr)}s of particle animation):")
    print(f"  samples: {[round(x, 2) for x in samples]}")
    print(f"  mean = {arr.mean():.2f}")
    print(f"  std  = {arr.std(ddof=1):.2f}")
    print(f"  min  = {arr.min():.2f}")
    print(f"  max  = {arr.max():.2f}")
    return web.Response(status=204)


@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)
    wslink_server.app.router.add_get("/borders/{z}/{x}/{y}.png", serve_border_tile)
    wslink_server.app.router.add_post("/bench", receive_bench)


# ============ TRAME MODULE: PARTICLE ANIMATION ============
# Register www/ as a trame module — serves static files and auto-loads particles.js

server.enable_module(
    {
        "serve": {"__wind": str(os.path.abspath("www"))},
        "scripts": ["__wind/particles.js"],
    }
)


# ============ UI LAYOUT ============

MOUSEMOVE_JS = """
    if ($event.buttons > 0) {
        tooltip_visible = false;
        return;
    }
    if (!window._ttThrottle || Date.now() - window._ttThrottle > 60) {
        window._ttThrottle = Date.now();

        var tiles = $event.currentTarget.querySelectorAll('.leaflet-tile');
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
            break;
        }
    }
"""

with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Speed")

    with layout.content:
        html_widgets.Style(
            """
            .leaflet-container {
                background: #1a1128 !important;
            }
            """
        )

        # Trame-native state watchers for particle control
        client.ClientStateChange(
            value="particle_active",
            change="""
                if (window._windParticles && window._windParticles.isReady()) {
                    value ? window._windParticles.start() : window._windParticles.stop();
                }
            """,
        )
        client.ClientStateChange(
            value="particle_count",
            change="""
                if (window._windParticles) {
                    window._windParticles.setCount(value);
                }
            """,
        )

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
                max_zoom=5,
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
            style=("`position:fixed; left:${tooltip_x}px; top:${tooltip_y - 45}px; "
                   "transform:translateX(-50%); background:rgba(30,30,30,0.88); color:white; "
                   "padding:5px 14px; border-radius:18px; font-size:13px; font-weight:500; "
                   "pointer-events:none; z-index:10000; white-space:nowrap; "
                   "backdrop-filter:blur(4px); font-family:-apple-system,BlinkMacSystemFont,sans-serif;`",),
        )

        # Legend
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
                            "rgb(91,70,168) 0%,"
                            "rgb(87,112,195) 6.25%,"
                            "rgb(95,155,207) 12.5%,"
                            "rgb(113,196,201) 18.75%,"
                            "rgb(133,225,174) 25%,"
                            "rgb(176,234,154) 31.25%,"
                            "rgb(228,242,145) 37.5%,"
                            "rgb(245,228,134) 43.75%,"
                            "rgb(248,192,123) 50%,"
                            "rgb(243,156,112) 56.25%,"
                            "rgb(231,127,105) 62.5%,"
                            "rgb(216,105,99) 68.75%,"
                            "rgb(196,88,99) 75%,"
                            "rgb(174,74,103) 81.25%,"
                            "rgb(155,62,106) 87.5%,"
                            "rgb(138,50,105) 93.75%,"
                            "rgb(122,39,106) 100%);"
                        ),
                    )
                with v3.VRow(no_gutters=True, style="margin-left:30px; width:280px;", justify="space-between"):
                    for val in ["0", "10", "20", "30", "40", "50", "60", "70", "80"]:
                        v3.VCardText(
                            val,
                            style="color:white; font-size:10px; padding:0; text-align:center; min-width:0;",
                        )

@server.controller.add("on_server_ready")
def on_ready(**kwargs):
    _t_ready = time.perf_counter()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"BENCHMARK: Server ready: {(_t_ready - _app_start)*1000:.1f}ms")
    print(f"BENCHMARK: Peak memory: {mem_mb:.1f} MB")

if __name__ == "__main__":
    server.start()
