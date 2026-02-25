"""
Wind visualization with pre-rendered tiles and VTK-rendered wind particles.
"""

import os
import json
import time
import math
import asyncio
import io
import base64
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, leaflet3 as leaflet, html as html_widgets

CACHE_BUST = int(time.time())

server = get_server(client_type="vue3")
state = server.state
state.trame__title = "Wind Speed"

# ============ WIND DATA LOOKUP ============

print("Loading wind data...")
with open('data/wind_data.json') as f:
    wind_data = json.load(f)

# Tooltip lookup grid (0.5deg resolution)
LOOKUP_STEP = 0.5
lookup_grid = {}

for p in wind_data:
    lat_key = round(round(p['lat'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    lon_key = round(round(p['lon'] / LOOKUP_STEP) * LOOKUP_STEP, 1)
    key = (lat_key, lon_key)
    if key not in lookup_grid:
        lookup_grid[key] = (round(p['speed'], 1), round(p['direction']))

del wind_data
print(f"Tooltip grid: {len(lookup_grid)} points")

# ============ WIND FIELD FOR PARTICLES (from NPZ) ============

print("Loading wind grid for particles...")
_npz = np.load('data/wind_grid.npz')
_full_lats = _npz['lats']  # (2881,) 90 -> -90, step 0.0625
_full_lons = _npz['lons']  # (5760,) -180 -> 179.9375, step 0.0625
_full_u = _npz['u']        # (2881, 5760)
_full_v = _npz['v']        # (2881, 5760)

# Subsample every 16th point (0.0625 * 16 = 1.0 degree)
WIND_STEP = 16
WIND_LATS = _full_lats[::WIND_STEP]
WIND_LONS = _full_lons[::WIND_STEP]
WIND_U = _full_u[::WIND_STEP, ::WIND_STEP].astype(np.float64)
WIND_V = _full_v[::WIND_STEP, ::WIND_STEP].astype(np.float64)
WIND_LAT_MAX = float(WIND_LATS[0])     # 90
WIND_LON_MIN = float(WIND_LONS[0])     # -180
WIND_LAT_STEP = abs(float(WIND_LATS[1] - WIND_LATS[0]))
WIND_LON_STEP = abs(float(WIND_LONS[1] - WIND_LONS[0]))

del _npz, _full_lats, _full_lons, _full_u, _full_v
print(f"Wind grid: {WIND_U.shape} at {WIND_LAT_STEP:.2f}deg resolution")

COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
           'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


def lookup_wind_tooltip(lat, lon):
    lat_key = round(round(lat / LOOKUP_STEP) * LOOKUP_STEP, 1)
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    lon_key = round(round(lon / LOOKUP_STEP) * LOOKUP_STEP, 1)
    return lookup_grid.get((lat_key, lon_key))


# ============ VTK PARTICLE RENDERER ============

class ParticleRenderer:
    """Off-screen VTK renderer for particle line segments."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.SetSize(width, height)
        self.render_window.SetAlphaBitPlanes(1)
        self.render_window.SetMultiSamples(0)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)
        self.renderer.SetBackgroundAlpha(0.0)
        self.render_window.AddRenderer(self.renderer)

        self.points = vtk.vtkPoints()
        self.lines = vtk.vtkCellArray()
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetLines(self.lines)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(self.polydata)

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.actor.GetProperty().SetOpacity(0.6)
        self.actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(self.actor)

        self._setup_camera()

        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(self.render_window)
        self.w2i.SetInputBufferTypeToRGBA()
        self.w2i.ReadFrontBufferOff()

    def _setup_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(True)
        camera.SetPosition(self.width / 2, self.height / 2, 1)
        camera.SetFocalPoint(self.width / 2, self.height / 2, 0)
        camera.SetParallelScale(self.height / 2)
        self.renderer.ResetCameraClippingRange()

    def resize(self, width, height):
        self.width = width
        self.height = height
        self.render_window.SetSize(width, height)
        self._setup_camera()

    def render_frame(self, segments):
        """
        Render line segments and return RGBA numpy array.
        segments: Nx4 array of (x0, y0, x1, y1) in pixel coords.
        Returns: (height, width, 4) uint8 array.
        """
        self.points.Reset()
        self.lines.Reset()

        n = len(segments)
        if n == 0:
            return np.zeros((self.height, self.width, 4), dtype=np.uint8)

        self.points.SetNumberOfPoints(n * 2)
        for i in range(n):
            # VTK Y is bottom-up, so flip Y coordinates
            self.points.SetPoint(i * 2, segments[i, 0], self.height - segments[i, 1], 0)
            self.points.SetPoint(i * 2 + 1, segments[i, 2], self.height - segments[i, 3], 0)

        for i in range(n):
            self.lines.InsertNextCell(2)
            self.lines.InsertCellPoint(i * 2)
            self.lines.InsertCellPoint(i * 2 + 1)

        self.polydata.Modified()
        self.render_window.Render()
        self.w2i.Modified()
        self.w2i.Update()

        output = self.w2i.GetOutput()
        dims = output.GetDimensions()
        rgba = vtk_to_numpy(output.GetPointData().GetScalars())
        rgba = rgba.reshape(dims[1], dims[0], -1)
        return np.flipud(rgba[:, :, :4].copy())


# ============ PARTICLE SIMULATION ============

class ParticleSimulation:
    """NumPy-vectorized particle advection through wind field."""

    NUM_PARTICLES = 1500
    MAX_AGE = 80
    SPEED_MAX_PX = 1.5
    MAX_WIND = 35.0

    def __init__(self):
        self.x = np.zeros(self.NUM_PARTICLES)
        self.y = np.zeros(self.NUM_PARTICLES)
        self.age = np.zeros(self.NUM_PARTICLES, dtype=np.int32)
        self.max_age = np.zeros(self.NUM_PARTICLES, dtype=np.int32)
        self.width = 1200
        self.height = 800
        self.zoom = 3
        self.center_lat = 20.0
        self.center_lon = 0.0
        self._initialized = False

    def init_particles(self):
        self.x = np.random.rand(self.NUM_PARTICLES) * self.width
        self.y = np.random.rand(self.NUM_PARTICLES) * self.height
        self.age = np.random.randint(0, self.MAX_AGE, self.NUM_PARTICLES)
        self.max_age = self.MAX_AGE + np.random.randint(0, 20, self.NUM_PARTICLES)
        self._initialized = True

    def update_viewport(self, zoom, center_lat, center_lon, width, height):
        self.zoom = zoom
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.width = width
        self.height = height
        self.init_particles()

    def pixel_to_latlon(self, px, py):
        """Convert screen pixel coords to lat/lon (vectorized, Web Mercator)."""
        total_pixels = 256.0 * (2 ** self.zoom)

        # Center of map in world pixel coordinates
        cx_world = (self.center_lon + 180.0) / 360.0 * total_pixels
        lat_rad = math.radians(self.center_lat)
        cy_world = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * total_pixels

        # Screen pixel -> world pixel
        wx = cx_world + (px - self.width / 2.0)
        wy = cy_world + (py - self.height / 2.0)

        # World pixel -> lat/lon (inverse Web Mercator)
        lon = wx / total_pixels * 360.0 - 180.0
        merc_y = math.pi * (1.0 - 2.0 * wy / total_pixels)
        lat = np.degrees(np.arctan(np.sinh(merc_y)))

        return lat, lon

    def lookup_wind_field(self, lats, lons):
        """Bilinear interpolation of wind vectors (vectorized)."""
        lons = np.where(lons > 180, lons - 360, lons)
        lons = np.where(lons < -180, lons + 360, lons)

        # Grid indices (lats descending: index 0 = WIND_LAT_MAX)
        li = (WIND_LAT_MAX - lats) / WIND_LAT_STEP
        lj = (lons - WIND_LON_MIN) / WIND_LON_STEP

        i0 = np.clip(np.floor(li).astype(int), 0, WIND_U.shape[0] - 2)
        j0 = np.clip(np.floor(lj).astype(int), 0, WIND_U.shape[1] - 2)

        fi = np.clip(li - i0, 0.0, 1.0)
        fj = np.clip(lj - j0, 0.0, 1.0)

        u = (WIND_U[i0, j0] * (1 - fi) * (1 - fj) +
             WIND_U[i0 + 1, j0] * fi * (1 - fj) +
             WIND_U[i0, j0 + 1] * (1 - fi) * fj +
             WIND_U[i0 + 1, j0 + 1] * fi * fj)

        v = (WIND_V[i0, j0] * (1 - fi) * (1 - fj) +
             WIND_V[i0 + 1, j0] * fi * (1 - fj) +
             WIND_V[i0, j0 + 1] * (1 - fi) * fj +
             WIND_V[i0 + 1, j0 + 1] * fi * fj)

        return u, v

    def step(self):
        """Advance particles one step. Returns Nx4 segment array."""
        if not self._initialized:
            return np.empty((0, 4))

        lats, lons = self.pixel_to_latlon(self.x, self.y)
        u, v = self.lookup_wind_field(lats, lons)

        speed_scale = self.SPEED_MAX_PX / self.MAX_WIND * (2 ** (self.zoom - 3))
        dx = u * speed_scale
        dy = -v * speed_scale  # Screen Y is inverted

        old_x = self.x.copy()
        old_y = self.y.copy()
        self.x += dx
        self.y += dy
        self.age += 1

        # Build segments for particles that moved
        moved = (np.abs(dx) > 0.01) | (np.abs(dy) > 0.01)
        if moved.any():
            segments = np.column_stack([
                old_x[moved], old_y[moved],
                self.x[moved], self.y[moved]
            ])
        else:
            segments = np.empty((0, 4))

        # Respawn dead or out-of-bounds particles
        dead = ((self.age > self.max_age) |
                (self.x < -10) | (self.x > self.width + 10) |
                (self.y < -10) | (self.y > self.height + 10))
        n_dead = dead.sum()
        if n_dead > 0:
            self.x[dead] = np.random.rand(n_dead) * self.width
            self.y[dead] = np.random.rand(n_dead) * self.height
            self.age[dead] = np.random.randint(0, self.MAX_AGE, n_dead)
            self.max_age[dead] = self.MAX_AGE + np.random.randint(0, 20, n_dead)

        return segments


# ============ TRAIL COMPOSITING ============

class TrailCompositor:
    """Maintains a fading trail buffer and encodes frames as base64 PNG."""

    def __init__(self, width, height, fade_alpha=0.95):
        self.fade_alpha = fade_alpha
        self.trail = np.zeros((height, width, 4), dtype=np.float32)

    def composite(self, new_frame_rgba):
        """Fade trails, add new segments, return base64 PNG data URI."""
        self.trail[:, :, 3] *= self.fade_alpha

        mask = new_frame_rgba[:, :, 3] > 0
        self.trail[mask] = new_frame_rgba[mask].astype(np.float32)

        frame_uint8 = np.clip(self.trail, 0, 255).astype(np.uint8)
        img = Image.fromarray(frame_uint8, 'RGBA')
        buf = io.BytesIO()
        img.save(buf, format='PNG', compress_level=1)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{b64}"

    def clear(self):
        self.trail[:] = 0

    def resize(self, width, height):
        self.trail = np.zeros((height, width, 4), dtype=np.float32)


# ============ TRAME STATE ============

state.tooltip_visible = False
state.tooltip_text = ""
state.tooltip_dir = ""
state.tooltip_arrow_rot = 0
state.tooltip_x = 0
state.tooltip_y = 0
state.mouse_data = None

state.particle_frame = ""
state.particle_active = True
state.map_viewport = None

simulation = ParticleSimulation()
particle_renderer = ParticleRenderer(1200, 800)
compositor = TrailCompositor(1200, 800)


@state.change("mouse_data")
def on_mouse_move(mouse_data, **kwargs):
    if mouse_data is None:
        state.tooltip_visible = False
        return
    try:
        lat, lon, cx, cy = mouse_data
        result = lookup_wind_tooltip(lat, lon)
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


@state.change("map_viewport")
def on_viewport_change(map_viewport, **kwargs):
    """Handle initial viewport dimensions from client JS."""
    if map_viewport is None:
        return
    try:
        zoom = int(map_viewport['zoom'])
        center = map_viewport['center']
        width = int(map_viewport['width'])
        height = int(map_viewport['height'])
        if width < 10 or height < 10:
            return
        simulation.update_viewport(zoom, center[0], center[1], width, height)
        particle_renderer.resize(width, height)
        compositor.resize(width, height)
    except Exception:
        pass


_viewport_update_task = None
_map_move_paused = False


@state.change("zoom", "center")
def on_map_move(zoom, center, **kwargs):
    """Debounced handler for map pan/zoom — pauses particles, resyncs after 400ms."""
    global _viewport_update_task, _map_move_paused
    if not simulation._initialized:
        return

    # Pause animation (only set state once to avoid flooding)
    if not _map_move_paused:
        _map_move_paused = True
        state.particle_active = False

    # Cancel any pending viewport update
    if _viewport_update_task is not None:
        _viewport_update_task.cancel()

    # Capture current values for the closure
    _zoom = int(zoom)
    _center = list(center)

    async def delayed_update():
        global _map_move_paused
        await asyncio.sleep(0.4)
        simulation.update_viewport(
            _zoom, _center[0], _center[1],
            simulation.width, simulation.height,
        )
        compositor.clear()
        _map_move_paused = False
        with state:
            state.particle_active = True

    _viewport_update_task = asyncio.ensure_future(delayed_update())


# ============ ANIMATION LOOP ============

TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS


async def animation_loop():
    """Server-side particle animation driven by VTK rendering.

    Note: VTK's Cocoa render window on macOS requires the main thread,
    so we cannot offload rendering to a thread pool. Instead we yield
    to the event loop between frames with asyncio.sleep.
    """
    while not simulation._initialized:
        await asyncio.sleep(0.5)

    print("Particle animation started")
    frame_count = 0
    t_start = time.time()

    while True:
        if not state.particle_active:
            await asyncio.sleep(0.1)
            continue

        t0 = time.time()

        segments = simulation.step()
        new_frame = particle_renderer.render_frame(segments)
        frame_uri = compositor.composite(new_frame)

        with state:
            state.particle_frame = frame_uri

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_total = time.time() - t_start
            avg_fps = frame_count / elapsed_total
            frame_ms = (time.time() - t0) * 1000
            print(f"Particles: {frame_count} frames, avg {avg_fps:.1f} FPS, last frame {frame_ms:.0f}ms")

        elapsed = time.time() - t0
        sleep_time = max(0.01, FRAME_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)


@server.controller.add("on_server_ready")
def start_animation(**kwargs):
    # Start with default viewport so animation begins immediately
    # (will be updated when client sends actual viewport dimensions)
    if not simulation._initialized:
        simulation.update_viewport(3, 20.0, 0.0, 1200, 800)
        print("Initialized with default viewport (1200x800, zoom 3)")
    asyncio.ensure_future(animation_loop())


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


@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)
    wslink_server.app.router.add_get("/borders/{z}/{x}/{y}.png", serve_border_tile)


# ============ UI LAYOUT ============

# Send viewport info on first mouse enter
INIT_VIEWPORT_JS = (
    "if(typeof window!=='undefined'&&!window._vpInit){"
    "window._vpInit=true;"
    "var ct=$event.currentTarget;"
    "window.setTimeout(function(){"
    "var el=ct?ct.querySelector('.leaflet-container'):null;"
    "if(el){var r=el.getBoundingClientRect();"
    "map_viewport={zoom:zoom,center:center,"
    "width:Math.round(r.width),height:Math.round(r.height)};}"
    "},1500);}"
)

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
            mouseenter=INIT_VIEWPORT_JS,
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

            # VTK-rendered particle overlay
            html_widgets.Img(
                src=("particle_frame",),
                style=(
                    "position:absolute; top:0; left:0; "
                    "width:100%; height:100%; "
                    "pointer-events:none; z-index:450;"
                ),
                v_show=("particle_frame",),
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
