"""
Pure VTK wind visualization benchmark — no trame, no web server.
Composites tiles into a background image, animates wind particles.
Matches trame version functionality: vectorized particles, FOV-based rendering,
pan/zoom with particle hide/show.
Prints FPS metrics for comparison with the trame version.
"""

import os
import json
import time
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk

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
    """Vectorized pixel -> lat/lon conversion."""
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
            pts = np.zeros((n_seg * 2, 3), dtype=np.float32)
            pts[0::2, 0] = segments[:, 0]
            pts[0::2, 1] = segments[:, 1]
            pts[1::2, 0] = segments[:, 2]
            pts[1::2, 1] = segments[:, 3]
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
        connectivity[:, 0] = 2
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

# Single renderer — matches trame version
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
window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(1024, 1024)
window.SetWindowName("VTK Wind Benchmark")

# Interactor with pan/zoom (no window/level)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
style = vtk.vtkInteractorStyleImage()
interactor.SetInteractorStyle(style)


# Override left-click from window/level (changes colors) to pan
def _left_down(obj, event):
    obj.OnMiddleButtonDown()


def _left_up(obj, event):
    obj.OnMiddleButtonUp()


style.AddObserver("LeftButtonPressEvent", _left_down)
style.AddObserver("LeftButtonReleaseEvent", _left_up)


# ============ CAMERA BOUNDS ============

def get_camera_bounds():
    """Get visible pixel bounds (x0, x1, y0, y1) from the orthographic camera."""
    fp = cam.GetFocalPoint()
    ps = cam.GetParallelScale()
    win_size = window.GetSize()
    aspect = win_size[0] / max(win_size[1], 1)
    half_h = ps
    half_w = ps * aspect
    return (fp[0] - half_w, fp[0] + half_w, fp[1] - half_h, fp[1] + half_h)


# ============ INTERACTION STATE ============

animating = True
interaction_end_time = None
RESPAWN_DELAY = 0.4


def on_interaction_start(obj=None, event=None):
    """Hide particles and stop animation during pan/zoom."""
    global animating, interaction_end_time
    animating = False
    interaction_end_time = None
    particle_actor.VisibilityOff()
    particles.trails.clear()


def on_interaction_end(obj=None, event=None):
    """Mark interaction end time — respawn happens in timer callback after delay."""
    global interaction_end_time
    interaction_end_time = time.perf_counter()


interactor.AddObserver("StartInteractionEvent", on_interaction_start)
interactor.AddObserver("EndInteractionEvent", on_interaction_end)

# ============ ANIMATION LOOP ============

frame_count = 0
total_frames = 0
fps_start = time.perf_counter()
last_report = fps_start
frame_times = []

# Set initial bounds
bounds = get_camera_bounds()
particles.set_bounds(bounds)
particles.respawn_all()


def on_timer(obj, event):
    global frame_count, total_frames, last_report
    global animating, interaction_end_time

    # Check for delayed respawn after interaction
    if interaction_end_time is not None:
        if time.perf_counter() - interaction_end_time >= RESPAWN_DELAY:
            interaction_end_time = None
            new_bounds = get_camera_bounds()
            particles.set_bounds(new_bounds)
            particles.respawn_all()
            particle_actor.VisibilityOn()
            animating = True

    if not animating:
        return

    t0 = time.perf_counter()

    # Update bounds from camera each frame
    current_bounds = get_camera_bounds()
    particles.set_bounds(current_bounds)

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
print("Pan: left-click drag | Zoom: scroll | Close window for final stats.\n")

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
