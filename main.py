"""
Wind visualization with pre-rendered tiles and animated wind particles.
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

CACHE_BUST = int(time.time())

server = get_server(client_type="vue3")
state = server.state
state.trame__title = "Wind Speed"

# ============ WIND DATA LOOKUP + WIND FIELD ============

print("Loading wind data...")
with open('data/wind_data.json') as f:
    wind_data = json.load(f)

# Tooltip lookup grid (0.5° resolution)
LOOKUP_STEP = 0.5
lookup_grid = {}

# Wind field for particle animation (1° resolution)
FIELD_STEP = 1.0
field_u = {}
field_v = {}

for p in wind_data:
    # Tooltip
    lat_key = round(round(p['lat'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    lon_key = round(round(p['lon'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    key = (lat_key, lon_key)
    if key not in lookup_grid:
        lookup_grid[key] = (round(p['speed'], 1), round(p['direction']))

    # Wind field
    flat = float(int(round(p['lat'] / FIELD_STEP))) * FIELD_STEP
    flon = float(int(round(p['lon'] / FIELD_STEP))) * FIELD_STEP
    fkey = (flat, flon)
    if fkey not in field_u:
        field_u[fkey] = round(p['u'], 2)
        field_v[fkey] = round(p['v'], 2)

print(f"Tooltip grid: {len(lookup_grid)} points")
print(f"Wind field grid: {len(field_u)} points")

# Build wind field JSON (served to client for particle animation)
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
print(f"Wind field JSON: {len(wind_field_json) / 1024:.0f} KB")

del wind_data, field_u, field_v, u_arr, v_arr

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
        return response
    return web.Response(status=404)


async def serve_wind_field(request):
    return web.Response(
        text=wind_field_json,
        content_type='application/json',
        headers={'Cache-Control': 'max-age=3600'}
    )


async def serve_particles_js(request):
    return web.Response(
        text=PARTICLES_JS,
        content_type='application/javascript'
    )


@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)
    wslink_server.app.router.add_get("/borders/{z}/{x}/{y}.png", serve_border_tile)
    wslink_server.app.router.add_get("/wind_field.json", serve_wind_field)
    wslink_server.app.router.add_get("/particles.js", serve_particles_js)


# ============ PARTICLE ANIMATION JS ============

PARTICLES_JS = """
(function() {
    'use strict';

    // === CONFIG ===
    var NUM_PARTICLES = 1500;
    var MAX_AGE = 80;
    var FADE_ALPHA = 0.95;
    var LINE_WIDTH = 3.0;
    var LINE_ALPHA = 0.6;
    var SPEED_MAX_PX = 1.5;
    var MAX_WIND = 35;

    // === STATE ===
    var canvas, ctx, trailCanvas, trailCtx;
    var particles = [];
    var windField = null;
    var mapState = null;
    var animating = false;
    var moveTimer = null;
    var animFrame = null;
    var container = null;

    // === INIT ===
    function init() {
        container = document.querySelector('.leaflet-container');
        if (!container) { setTimeout(init, 500); return; }

        fetch('/wind_field.json')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                windField = data;
                createCanvas();
                attachListeners();
                startAnimation();
            });
    }

    function createCanvas() {
        var dpr = window.devicePixelRatio || 1;
        var w = container.clientWidth;
        var h = container.clientHeight;

        canvas = document.createElement('canvas');
        canvas.id = 'wind-particles';
        canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:450;';
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        container.appendChild(canvas);
        ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        trailCanvas = document.createElement('canvas');
        trailCanvas.width = w * dpr;
        trailCanvas.height = h * dpr;
        trailCtx = trailCanvas.getContext('2d');
        trailCtx.scale(dpr, dpr);
    }

    function resizeCanvas() {
        if (!canvas || !container) return;
        var dpr = window.devicePixelRatio || 1;
        var w = container.clientWidth;
        var h = container.clientHeight;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);
        trailCanvas.width = w * dpr;
        trailCanvas.height = h * dpr;
        trailCtx.scale(dpr, dpr);
    }

    // === LISTENERS ===
    function attachListeners() {
        container.addEventListener('mousedown', onMoveStart, true);
        container.addEventListener('touchstart', onMoveStart, true);
        container.addEventListener('wheel', function() {
            onMoveStart();
            clearTimeout(moveTimer);
            moveTimer = setTimeout(onMoveEnd, 600);
        }, true);
        document.addEventListener('mouseup', onMoveEnd, true);
        document.addEventListener('touchend', onMoveEnd, true);
        window.addEventListener('resize', function() {
            resizeCanvas();
            onMoveStart();
            clearTimeout(moveTimer);
            moveTimer = setTimeout(onMoveEnd, 500);
        });
    }

    function onMoveStart() {
        animating = false;
        if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
        if (ctx && container) {
            var w = container.clientWidth;
            var h = container.clientHeight;
            ctx.clearRect(0, 0, w, h);
            trailCtx.clearRect(0, 0, w, h);
        }
    }

    function onMoveEnd() {
        clearTimeout(moveTimer);
        moveTimer = setTimeout(function() {
            startAnimation();
        }, 400);
    }

    // === MAP PROJECTION ===
    function updateMapState() {
        var tiles = container.querySelectorAll('.leaflet-tile');
        var cr = container.getBoundingClientRect();
        for (var i = 0; i < tiles.length; i++) {
            var src = tiles[i].src || '';
            var m = src.match(/\\/tiles\\/(\\d+)\\/(\\d+)\\/(\\d+)\\.png/);
            if (!m) continue;
            var rect = tiles[i].getBoundingClientRect();
            if (rect.width < 10) continue;
            mapState = {
                n: Math.pow(2, parseInt(m[1])),
                zoom: parseInt(m[1]),
                tw: rect.width,
                rtx: parseInt(m[2]),
                rty: parseInt(m[3]),
                rpx: rect.left - cr.left,
                rpy: rect.top - cr.top
            };
            return true;
        }
        return false;
    }

    function pixelToLatLon(px, py) {
        var s = mapState;
        var tx = s.rtx + (px - s.rpx) / s.tw;
        var ty = s.rty + (py - s.rpy) / s.tw;
        var lon = tx / s.n * 360 - 180;
        var lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / s.n))) * 180 / Math.PI;
        return [lat, lon];
    }

    // === WIND LOOKUP ===
    function getWind(lat, lon) {
        if (!windField) return [0, 0];
        while (lon > 180) lon -= 360;
        while (lon < -180) lon += 360;

        var li = (windField.lat_max - lat) / windField.lat_step;
        var lj = (lon - windField.lon_min) / windField.lon_step;
        var i0 = Math.floor(li), j0 = Math.floor(lj);
        var i1 = i0 + 1, j1 = j0 + 1;

        if (i0 < 0 || i1 >= windField.n_lats || j0 < 0 || j1 >= windField.n_lons)
            return [0, 0];

        var fi = li - i0, fj = lj - j0;
        var nl = windField.n_lons;
        var u = windField.u, v = windField.v;

        var ui = u[i0*nl+j0]*(1-fi)*(1-fj) + u[i0*nl+j1]*(1-fi)*fj
               + u[i1*nl+j0]*fi*(1-fj) + u[i1*nl+j1]*fi*fj;
        var vi = v[i0*nl+j0]*(1-fi)*(1-fj) + v[i0*nl+j1]*(1-fi)*fj
               + v[i1*nl+j0]*fi*(1-fj) + v[i1*nl+j1]*fi*fj;

        return [ui, vi];
    }

    // === PARTICLES ===
    function spawnParticle() {
        var w = container.clientWidth, h = container.clientHeight;
        return {
            x: Math.random() * w,
            y: Math.random() * h,
            age: Math.floor(Math.random() * MAX_AGE),
            maxAge: MAX_AGE + Math.floor(Math.random() * 20)
        };
    }

    function startAnimation() {
        if (!updateMapState()) {
            setTimeout(startAnimation, 500);
            return;
        }
        particles = [];
        for (var i = 0; i < NUM_PARTICLES; i++) particles.push(spawnParticle());

        var w = container.clientWidth, h = container.clientHeight;
        ctx.clearRect(0, 0, w, h);
        trailCtx.clearRect(0, 0, w, h);
        animating = true;
        animate();
    }

    function animate() {
        if (!animating) return;

        var w = container.clientWidth, h = container.clientHeight;
        var speedScale = SPEED_MAX_PX / MAX_WIND * Math.pow(2, mapState.zoom - 3);

        // Fade trails
        trailCtx.clearRect(0, 0, w, h);
        trailCtx.drawImage(canvas, 0, 0, w, h);
        ctx.clearRect(0, 0, w, h);
        ctx.globalAlpha = FADE_ALPHA;
        ctx.drawImage(trailCanvas, 0, 0, w, h);
        ctx.globalAlpha = 1.0;

        // Draw new segments
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,' + LINE_ALPHA + ')';
        ctx.lineWidth = LINE_WIDTH;

        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            var ll = pixelToLatLon(p.x, p.y);
            if (!ll) { particles[i] = spawnParticle(); continue; }

            var wind = getWind(ll[0], ll[1]);
            var dx = wind[0] * speedScale;
            var dy = -wind[1] * speedScale;

            if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) { p.age++; }
            else {
                var nx = p.x + dx, ny = p.y + dy;
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(nx, ny);
                p.x = nx;
                p.y = ny;
            }

            p.age++;
            if (p.age > p.maxAge || p.x < -10 || p.x > w+10 || p.y < -10 || p.y > h+10) {
                particles[i] = spawnParticle();
            }
        }
        ctx.stroke();
        animFrame = requestAnimationFrame(animate);
    }

    // Start
    init();
})();
"""


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

        with v3.VContainer(
            fluid=True,
            classes="fill-height pa-0",
            style="position: relative;",
            mouseenter=(
                "if(!window._pLoaded){"
                "window._pLoaded=true;"
                "window.setTimeout(function(){"
                "var s=window.document.createElement('script');"
                "s.src='/particles.js';"
                "window.document.head.appendChild(s);},1000);}"
            ),
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

if __name__ == "__main__":
    server.start()