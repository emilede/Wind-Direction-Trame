"""
Wind visualization using trame + VTK only. No JavaScript. No Leaflet.
- Tiles composited into a VTK background image
- Wind particles simulated in Python/numpy, rendered as VTK polydata
- VtkRemoteView streams the rendered scene to the browser
- Trame UI for legend, controls, FPS display
"""

import os
import json
import time
import asyncio
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, html as html_widgets
from trame.widgets import vtk as vtk_widgets

# ============ CONFIG ============

ZOOM = 3
TILE_SIZE = 256
NUM_TILES = 2 ** ZOOM
IMG_W = NUM_TILES * TILE_SIZE
IMG_H = NUM_TILES * TILE_SIZE

NUM_PARTICLES = 800
MAX_AGE = 80
SPEED_SCALE = 0.6
MAX_WIND = 35
TRAIL_LENGTH = 6
LINE_WIDTH = 2.0
TARGET_FPS = 20
FRAME_INTERVAL = 1.0 / TARGET_FPS

# ============ TRAME SERVER ============

server = get_server(client_type="vue3")
state = server.state
ctrl = server.controller
state.trame__title = "Wind Speed — VTK"
state.fps_text = "FPS: --"

# ============ LOAD + COMPOSITE TILES ============

print("Compositing tiles...")
t0 = time.perf_counter()

bg = Image.new('RGB', (IMG_W, IMG_H), (26, 17, 40))
for x in range(NUM_TILES):
    for y in range(NUM_TILES):
        path = f"data/tiles/{ZOOM}/{x}/{y}.png"
        if os.path.exists(path):
            tile = Image.open(path)
            bg.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

for x in range(NUM_TILES):
    for y in range(NUM_TILES):
        path = f"data/border_tiles/{ZOOM}/{x}/{y}.png"
        if os.path.exists(path):
            border = Image.open(path).convert('RGBA')
            bg.paste(border, (x * TILE_SIZE, y * TILE_SIZE), border)

print(f"Tiles composited in {time.perf_counter() - t0:.2f}s")

# Convert to vtkImageData (flip for VTK's bottom-left origin)
bg_arr = np.flipud(np.array(bg))
bg_vtk = vtk.vtkImageData()
bg_vtk.SetDimensions(IMG_W, IMG_H, 1)
bg_vtk.SetSpacing(1.0, 1.0, 1.0)
bg_vtk.SetOrigin(0.0, 0.0, 0.0)
vtk_arr = numpy_to_vtk(bg_arr.reshape(-1, 3), deep=True)
vtk_arr.SetName("TileRGB")
bg_vtk.GetPointData().SetScalars(vtk_arr)

# ============ LOAD WIND DATA ============

print("Loading wind data...")
with open('data/wind_data.json') as f:
    wind_data = json.load(f)

FIELD_STEP = 1.0
field_u = {}
field_v = {}
for p in wind_data:
    flat = float(int(round(p['lat'] / FIELD_STEP))) * FIELD_STEP
    flon = float(int(round(p['lon'] / FIELD_STEP))) * FIELD_STEP
    fkey = (flat, flon)
    if fkey not in field_u:
        field_u[fkey] = p['u']
        field_v[fkey] = p['v']

N_LATS = 181
N_LONS = 361
u_grid = np.zeros((N_LATS, N_LONS), dtype=np.float32)
v_grid = np.zeros((N_LATS, N_LONS), dtype=np.float32)
for i, lat in enumerate(range(90, -91, -1)):
    for j, lon in enumerate(range(-180, 181, 1)):
        key = (float(lat), float(lon))
        u_grid[i, j] = field_u.get(key, 0.0)
        v_grid[i, j] = field_v.get(key, 0.0)

del wind_data, field_u, field_v
print(f"Wind field: {N_LATS}x{N_LONS}")


# ============ VECTORIZED PARTICLE SYSTEM ============

def pixel_to_latlon_vec(px, py):
    """Vectorized pixel → lat/lon conversion."""
    tx = px / TILE_SIZE
    ty = NUM_TILES - (py / TILE_SIZE)
    lon = tx / NUM_TILES * 360 - 180
    lat = np.arctan(np.sinh(np.pi * (1 - 2 * ty / NUM_TILES))) * 180 / np.pi
    return lat, lon


def get_wind_vec(lats, lons):
    """Vectorized bilinear wind interpolation."""
    lons = np.where(lons > 180, lons - 360, lons)
    lons = np.where(lons < -180, lons + 360, lons)

    li = (90.0 - lats) / FIELD_STEP
    lj = (lons + 180.0) / FIELD_STEP
    i0 = np.floor(li).astype(int)
    j0 = np.floor(lj).astype(int)
    i1 = i0 + 1
    j1 = j0 + 1

    valid = (i0 >= 0) & (i1 < N_LATS) & (j0 >= 0) & (j1 < N_LONS)
    i0c = np.clip(i0, 0, N_LATS - 1)
    j0c = np.clip(j0, 0, N_LONS - 1)
    i1c = np.clip(i1, 0, N_LATS - 1)
    j1c = np.clip(j1, 0, N_LONS - 1)

    fi = li - i0
    fj = lj - j0

    u = (u_grid[i0c, j0c] * (1 - fi) * (1 - fj) +
         u_grid[i0c, j1c] * (1 - fi) * fj +
         u_grid[i1c, j0c] * fi * (1 - fj) +
         u_grid[i1c, j1c] * fi * fj)
    v = (v_grid[i0c, j0c] * (1 - fi) * (1 - fj) +
         v_grid[i0c, j1c] * (1 - fi) * fj +
         v_grid[i1c, j0c] * fi * (1 - fj) +
         v_grid[i1c, j1c] * fi * fj)

    u = np.where(valid, u, 0.0)
    v = np.where(valid, v, 0.0)
    return u, v


class ParticleSystem:
    def __init__(self, n, w, h):
        self.n = n
        self.w = w
        self.h = h
        # Bounds: (x0, x1, y0, y1) — defaults to full image
        self.bounds = (0, w, 0, h)
        self.x = np.random.rand(n).astype(np.float32) * w
        self.y = np.random.rand(n).astype(np.float32) * h
        self.age = np.random.randint(0, MAX_AGE, n)
        self.max_age = MAX_AGE + np.random.randint(0, 20, n)
        self.trails = []

    def set_bounds(self, bounds):
        """Update visible bounds (x0, x1, y0, y1)."""
        self.bounds = bounds

    def _spawn_in_bounds(self, count):
        """Spawn particles within current visible bounds."""
        x0, x1, y0, y1 = self.bounds
        bw = max(x1 - x0, 1)
        bh = max(y1 - y0, 1)
        xs = np.random.rand(count).astype(np.float32) * bw + x0
        ys = np.random.rand(count).astype(np.float32) * bh + y0
        return xs, ys

    def respawn_all(self):
        """Reset all particles within current visible bounds."""
        self.x, self.y = self._spawn_in_bounds(self.n)
        self.age[:] = np.random.randint(0, MAX_AGE, self.n)
        self.trails.clear()

    def step(self):
        prev_x = self.x.copy()
        prev_y = self.y.copy()

        lats, lons = pixel_to_latlon_vec(self.x, self.y)
        u, v = get_wind_vec(lats, lons)

        dx = u * SPEED_SCALE
        dy = v * SPEED_SCALE
        moved = (np.abs(dx) > 0.01) | (np.abs(dy) > 0.01)

        self.x += dx
        self.y += dy
        self.age += 1

        # Respawn dead/oob particles — use visible bounds with margin
        x0, x1, y0, y1 = self.bounds
        margin = 10
        oob = ((self.age > self.max_age) |
               (self.x < x0 - margin) | (self.x > x1 + margin) |
               (self.y < y0 - margin) | (self.y > y1 + margin))
        n_respawn = np.sum(oob)
        if n_respawn > 0:
            self.x[oob], self.y[oob] = self._spawn_in_bounds(n_respawn)
            self.age[oob] = 0
            self.max_age[oob] = MAX_AGE + np.random.randint(0, 20, n_respawn)
            moved[oob] = False

        # Store trail segments
        idx = np.where(moved)[0]
        segments = np.column_stack([prev_x[idx], prev_y[idx], self.x[idx], self.y[idx]])
        self.trails.append(segments)
        if len(self.trails) > TRAIL_LENGTH:
            self.trails.pop(0)

    def build_polydata(self):
        """Build vtkPolyData from trail segments."""
        all_pts = []
        all_colors = []
        n_trails = len(self.trails)

        for t_i, segments in enumerate(self.trails):
            if len(segments) == 0:
                continue
            alpha = int(255 * ((t_i + 1) / n_trails) * 0.6)
            n_seg = len(segments)
            # Each segment = 2 points
            pts = np.zeros((n_seg * 2, 3), dtype=np.float32)
            pts[0::2, 0] = segments[:, 0]  # x0
            pts[0::2, 1] = segments[:, 1]  # y0
            pts[1::2, 0] = segments[:, 2]  # x1
            pts[1::2, 1] = segments[:, 3]  # y1
            pts[:, 2] = 0.1
            all_pts.append(pts)

            colors = np.full((n_seg, 4), [255, 255, 255, alpha], dtype=np.uint8)
            all_colors.append(colors)

        if not all_pts:
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk.vtkPoints())
            return polydata

        all_pts = np.concatenate(all_pts)
        all_colors = np.concatenate(all_colors)

        points = vtk.vtkPoints()
        vtk_points = numpy_to_vtk(all_pts, deep=True)
        points.SetData(vtk_points)

        n_lines = len(all_colors)
        cells = vtk.vtkCellArray()
        connectivity = np.zeros((n_lines, 3), dtype=np.int64)
        connectivity[:, 0] = 2  # 2 points per line
        connectivity[:, 1] = np.arange(0, n_lines * 2, 2)
        connectivity[:, 2] = np.arange(1, n_lines * 2 + 1, 2)
        vtk_cells = numpy_to_vtk(connectivity.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
        cells.SetCells(n_lines, vtk_cells)

        colors_vtk = numpy_to_vtk(all_colors, deep=True)
        colors_vtk.SetName("Colors")

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(cells)
        polydata.GetCellData().SetScalars(colors_vtk)
        return polydata


# ============ VTK SCENE ============

print("Setting up VTK scene...")

# Single renderer — both background and particles share a camera
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.07, 0.16)

bg_actor = vtk.vtkImageActor()
bg_actor.GetMapper().SetInputData(bg_vtk)
renderer.AddActor(bg_actor)

particles = ParticleSystem(NUM_PARTICLES, IMG_W, IMG_H)
particle_mapper = vtk.vtkPolyDataMapper()
particle_mapper.ScalarVisibilityOn()
particle_mapper.SetColorModeToDirectScalars()
particle_actor = vtk.vtkActor()
particle_actor.SetMapper(particle_mapper)
particle_actor.GetProperty().SetLineWidth(LINE_WIDTH)
renderer.AddActor(particle_actor)

# Camera — orthographic, framing the full tile image
cam = renderer.GetActiveCamera()
cam.ParallelProjectionOn()
cam.SetPosition(IMG_W / 2, IMG_H / 2, 1000)
cam.SetFocalPoint(IMG_W / 2, IMG_H / 2, 0)
cam.SetParallelScale(IMG_H / 2)

# Render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1024, 1024)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
style = vtk.vtkInteractorStyleImage()
interactor.SetInteractorStyle(style)

# Override left-click from window/level (changes colors) to pan
def _left_down(obj, event):
    obj.OnMiddleButtonDown()

def _left_up(obj, event):
    obj.OnMiddleButtonUp()

style.AddObserver("LeftButtonPressEvent", _left_down)
style.AddObserver("LeftButtonReleaseEvent", _left_up)

# Warmup render — initializes GL context before trame connects
render_window.Render()
print("Warmup render complete")


def get_camera_bounds():
    """Get visible pixel bounds (x0, x1, y0, y1) from the orthographic camera."""
    fp = cam.GetFocalPoint()
    ps = cam.GetParallelScale()
    win_size = render_window.GetSize()
    aspect = win_size[0] / max(win_size[1], 1)
    half_h = ps
    half_w = ps * aspect
    return (fp[0] - half_w, fp[0] + half_w, fp[1] - half_h, fp[1] + half_h)

# Track interaction state — hide particles during pan/zoom, respawn after
is_interacting = False
interaction_timer = None


# ============ ANIMATION LOOP ============

frame_count = 0
fps_start = time.perf_counter()
last_report = fps_start
animating = True


def on_interaction_start(obj=None, event=None):
    """Hide particles and stop animation during pan/zoom."""
    global animating, is_interacting, interaction_timer
    is_interacting = True
    animating = False
    particle_actor.VisibilityOff()
    particles.trails.clear()
    if interaction_timer:
        interaction_timer.cancel()


def on_interaction_end(obj=None, event=None):
    """Respawn particles after a brief delay when interaction stops."""
    global interaction_timer

    async def respawn():
        await asyncio.sleep(0.4)
        global animating, is_interacting
        is_interacting = False
        bounds = get_camera_bounds()
        particles.set_bounds(bounds)
        particles.respawn_all()
        particle_actor.VisibilityOn()
        animating = True

    interaction_timer = asyncio.ensure_future(respawn())


# Wire up interaction observers on the VTK interactor
interactor.AddObserver("StartInteractionEvent", on_interaction_start)
interactor.AddObserver("EndInteractionEvent", on_interaction_end)


async def animate():
    global frame_count, fps_start, last_report

    # Set initial bounds from camera
    bounds = get_camera_bounds()
    particles.set_bounds(bounds)
    particles.respawn_all()

    while True:
        t0 = time.perf_counter()

        if animating:
            # Update bounds from camera each frame (in case of slow drift)
            bounds = get_camera_bounds()
            particles.set_bounds(bounds)

            particles.step()
            polydata = particles.build_polydata()
            particle_mapper.SetInputData(polydata)
            particle_mapper.Update()
            ctrl.view_update()

            frame_count += 1
            now = time.perf_counter()
            if now - last_report >= 2.0:
                fps = frame_count / (now - last_report)
                state.fps_text = f"FPS: {fps:.1f}"
                print(f"FPS: {fps:.1f} | frames: {frame_count}")
                frame_count = 0
                last_report = now

        elapsed = time.perf_counter() - t0
        sleep_time = max(0.001, FRAME_INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)


@ctrl.add("on_server_ready")
def on_ready(**kwargs):
    asyncio.ensure_future(animate())


# ============ UI LAYOUT ============

with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Speed — VTK")

    with layout.content:
        html_widgets.Style(
            """
            html, body { margin: 0; padding: 0; overflow: hidden; }
            .v-application { background: #1a1128 !important; }
            """
        )

        with v3.VContainer(fluid=True, classes="fill-height pa-0"):
            view = vtk_widgets.VtkRemoteView(
                render_window,
                style="width: 100%; height: 100%;",
                interactive_ratio=1,
                still_ratio=1,
            )
            ctrl.view_update = view.update

        # FPS overlay
        html_widgets.Div(
            v_text=("fps_text",),
            style="position:fixed; top:10px; right:10px; color:white; "
                  "font-size:14px; font-weight:600; z-index:10000; "
                  "background:rgba(0,0,0,0.5); padding:4px 10px; border-radius:8px; "
                  "font-family:monospace;",
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
