"""
Pure VTK wind visualization benchmark — no trame, no web server.
Composites tiles into a background image, animates wind particles.
Prints FPS metrics for comparison with the trame version.
"""

import os
import json
import time
import math
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk

# ============ CONFIG ============

ZOOM = 3
TILE_SIZE = 256
NUM_TILES = 2 ** ZOOM  # 8
IMG_W = NUM_TILES * TILE_SIZE  # 2048
IMG_H = NUM_TILES * TILE_SIZE  # 2048

NUM_PARTICLES = 1500
MAX_AGE = 80
SPEED_SCALE = 1.5
MAX_WIND = 35
TRAIL_LENGTH = 6
LINE_WIDTH = 2.0

# ============ LOAD + COMPOSITE TILES ============

print("Loading tiles from disk...")
t_tile_start = time.perf_counter()

bg = Image.new('RGB', (IMG_W, IMG_H), (26, 17, 40))
tiles_loaded = 0
for x in range(NUM_TILES):
    for y in range(NUM_TILES):
        path = f"data/tiles/{ZOOM}/{x}/{y}.png"
        if os.path.exists(path):
            tile = Image.open(path)
            bg.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
            tiles_loaded += 1

for x in range(NUM_TILES):
    for y in range(NUM_TILES):
        path = f"data/border_tiles/{ZOOM}/{x}/{y}.png"
        if os.path.exists(path):
            border = Image.open(path).convert('RGBA')
            bg.paste(border, (x * TILE_SIZE, y * TILE_SIZE), border)
            tiles_loaded += 1

t_tile_end = time.perf_counter()
tile_load_time = t_tile_end - t_tile_start
print(f"Loaded {tiles_loaded} tiles in {tile_load_time:.2f}s")

# Convert PIL image to vtkImageData
# PIL is top-left origin, VTK is bottom-left origin — flip vertically
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


# ============ COORDINATE CONVERSION ============
# VTK coordinate system: origin bottom-left, y goes up
# Tile coordinate system: origin top-left, y goes down
# We flipped the image with np.flipud, so VTK pixel (px, py) where py=0 is
# the BOTTOM of the image = the BOTTOM of the tile grid (y = NUM_TILES)

def pixel_to_latlon(px, py):
    """Convert VTK pixel coords to lat/lon. py=0 is bottom, py=IMG_H is top."""
    # tile x coordinate (left to right)
    tx = px / TILE_SIZE
    # tile y coordinate — VTK y=0 is bottom of image = tile y=NUM_TILES
    ty = NUM_TILES - (py / TILE_SIZE)
    lon = tx / NUM_TILES * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / NUM_TILES)))
    lat = lat_rad * 180 / math.pi
    return lat, lon


def get_wind(lat, lon):
    """Bilinear interpolation of wind at lat/lon."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360

    li = (90.0 - lat) / FIELD_STEP
    lj = (lon + 180.0) / FIELD_STEP
    i0 = int(math.floor(li))
    j0 = int(math.floor(lj))
    i1 = i0 + 1
    j1 = j0 + 1

    if i0 < 0 or i1 >= N_LATS or j0 < 0 or j1 >= N_LONS:
        return 0.0, 0.0

    fi = li - i0
    fj = lj - j0
    u = (u_grid[i0, j0] * (1 - fi) * (1 - fj) +
         u_grid[i0, j1] * (1 - fi) * fj +
         u_grid[i1, j0] * fi * (1 - fj) +
         u_grid[i1, j1] * fi * fj)
    v = (v_grid[i0, j0] * (1 - fi) * (1 - fj) +
         v_grid[i0, j1] * (1 - fi) * fj +
         v_grid[i1, j0] * fi * (1 - fj) +
         v_grid[i1, j1] * fi * fj)
    return u, v


# ============ PARTICLE SYSTEM ============

class ParticleSystem:
    def __init__(self, n, w, h):
        self.n = n
        self.w = w
        self.h = h
        self.x = np.random.rand(n) * w
        self.y = np.random.rand(n) * h
        self.age = np.random.randint(0, MAX_AGE, n)
        self.max_age = MAX_AGE + np.random.randint(0, 20, n)
        self.trails = []

    def step(self):
        prev_x = self.x.copy()
        prev_y = self.y.copy()
        moved = np.zeros(self.n, dtype=bool)

        for i in range(self.n):
            lat, lon = pixel_to_latlon(self.x[i], self.y[i])
            u, v = get_wind(lat, lon)
            # u = east component, v = north component
            # In VTK: x is east, y is up (north)
            dx = u * SPEED_SCALE
            dy = v * SPEED_SCALE

            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.x[i] += dx
                self.y[i] += dy
                moved[i] = True

            self.age[i] += 1

            if (self.age[i] > self.max_age[i] or
                    self.x[i] < -10 or self.x[i] > self.w + 10 or
                    self.y[i] < -10 or self.y[i] > self.h + 10):
                self.x[i] = np.random.rand() * self.w
                self.y[i] = np.random.rand() * self.h
                self.age[i] = 0
                self.max_age[i] = MAX_AGE + np.random.randint(0, 20)
                moved[i] = False

        segments = []
        for i in range(self.n):
            if moved[i]:
                segments.append((prev_x[i], prev_y[i], self.x[i], self.y[i]))

        self.trails.append(segments)
        if len(self.trails) > TRAIL_LENGTH:
            self.trails.pop(0)

    def build_polydata(self):
        """Build vtkPolyData with all trail segments."""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4)
        colors.SetName("Colors")

        pt_idx = 0
        n_trails = len(self.trails)
        for t_i, segments in enumerate(self.trails):
            alpha = int(255 * ((t_i + 1) / n_trails) * 0.6)
            for x0, y0, x1, y1 in segments:
                points.InsertNextPoint(x0, y0, 0.1)
                points.InsertNextPoint(x1, y1, 0.1)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, pt_idx)
                line.GetPointIds().SetId(1, pt_idx + 1)
                lines.InsertNextCell(line)
                colors.InsertNextTuple4(255, 255, 255, alpha)
                pt_idx += 2

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetCellData().SetScalars(colors)
        return polydata


# ============ VTK SETUP ============

print("Setting up VTK...")

# --- Background renderer (layer 0) ---
bg_renderer = vtk.vtkRenderer()
bg_renderer.SetLayer(0)
bg_renderer.InteractiveOff()

bg_actor = vtk.vtkImageActor()
bg_actor.GetMapper().SetInputData(bg_vtk)
bg_renderer.AddActor(bg_actor)

# Set camera to frame the image exactly
bg_cam = bg_renderer.GetActiveCamera()
bg_cam.ParallelProjectionOn()
bg_cam.SetPosition(IMG_W / 2, IMG_H / 2, 1000)
bg_cam.SetFocalPoint(IMG_W / 2, IMG_H / 2, 0)
bg_cam.SetParallelScale(IMG_H / 2)

# --- Particle renderer (layer 1, transparent background) ---
fg_renderer = vtk.vtkRenderer()
fg_renderer.SetLayer(1)
fg_renderer.SetBackground(0, 0, 0)

particles = ParticleSystem(NUM_PARTICLES, IMG_W, IMG_H)
particle_mapper = vtk.vtkPolyDataMapper()
particle_mapper.ScalarVisibilityOn()
particle_mapper.SetColorModeToDirectScalars()
particle_actor = vtk.vtkActor()
particle_actor.SetMapper(particle_mapper)
particle_actor.GetProperty().SetLineWidth(LINE_WIDTH)
fg_renderer.AddActor(particle_actor)

# Match cameras
fg_cam = fg_renderer.GetActiveCamera()
fg_cam.ParallelProjectionOn()
fg_cam.SetPosition(IMG_W / 2, IMG_H / 2, 1000)
fg_cam.SetFocalPoint(IMG_W / 2, IMG_H / 2, 0)
fg_cam.SetParallelScale(IMG_H / 2)

# --- Render window ---
window = vtk.vtkRenderWindow()
window.SetNumberOfLayers(2)
window.AddRenderer(bg_renderer)
window.AddRenderer(fg_renderer)
window.SetSize(1024, 1024)
window.SetWindowName("VTK Wind Benchmark")

# --- Interactor (locked camera — benchmark only, no pan/zoom) ---
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
style = vtk.vtkInteractorStyleUser()  # disables all default interaction
interactor.SetInteractorStyle(style)

# ============ ANIMATION LOOP ============

frame_count = 0
total_frames = 0
fps_start = time.perf_counter()
last_report = fps_start
frame_times = []


def on_timer(obj, event):
    global frame_count, total_frames, last_report

    t0 = time.perf_counter()

    particles.step()
    polydata = particles.build_polydata()
    particle_mapper.SetInputData(polydata)
    window.Render()

    t1 = time.perf_counter()
    frame_times.append(t1 - t0)

    frame_count += 1
    total_frames += 1

    now = time.perf_counter()
    if now - last_report >= 2.0:
        fps = frame_count / (now - last_report)
        avg_ms = np.mean(frame_times[-frame_count:]) * 1000
        print(f"FPS: {fps:.1f} | avg frame: {avg_ms:.1f}ms | total frames: {total_frames}")
        frame_count = 0
        last_report = now


interactor.AddObserver('TimerEvent', on_timer)
timer_id = interactor.CreateRepeatingTimer(16)

print(f"\nStarting VTK benchmark: {NUM_PARTICLES} particles, {TRAIL_LENGTH} trail frames")
print("Close the window to see final stats.\n")

interactor.Initialize()

t_first_render = time.perf_counter()
window.Render()
first_render_time = time.perf_counter() - t_first_render

print(f"First render: {first_render_time * 1000:.0f}ms")

interactor.Start()

# ============ FINAL STATS ============
elapsed = time.perf_counter() - fps_start
if total_frames > 0:
    print(f"\n{'='*50}")
    print(f"  VTK STANDALONE BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"  Tile loading:     {tile_load_time:.2f}s ({tiles_loaded} tiles from disk)")
    print(f"  First render:     {first_render_time * 1000:.0f}ms")
    print(f"  Time to visible:  {tile_load_time + first_render_time:.2f}s")
    print(f"{'='*50}")
    print(f"  Animation FPS:    {total_frames / elapsed:.1f} avg")
    print(f"  Total frames:     {total_frames}")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"  Avg frame time:   {np.mean(frame_times) * 1000:.1f}ms")
    print(f"  Min frame time:   {np.min(frame_times) * 1000:.1f}ms")
    print(f"  Max frame time:   {np.max(frame_times) * 1000:.1f}ms")
    print(f"  P95 frame time:   {np.percentile(frame_times, 95) * 1000:.1f}ms")
    print(f"{'='*50}")
