"""
Wind visualization with pre-rendered tiles.
"""

import os
import time
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, leaflet3 as leaflet

# Cache buster - change this or use timestamp
CACHE_BUST = int(time.time())

server = get_server(client_type="vue3")
state = server.state
state.trame__title = "Wind Speed"

# Tile serving handler
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

# Border tile serving handler
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

# Register routes after server is ready
@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)
    wslink_server.app.router.add_get("/borders/{z}/{x}/{y}.png", serve_border_tile)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Speed")
    
    with layout.content:
        with v3.VContainer(fluid=True, classes="fill-height pa-0"):
            with leaflet.LMap(
                zoom=("zoom", 2),
                center=("center", [20, 0]),
                world_copy_jump=True,
                max_zoom=3,
                min_zoom=2,
                style="height: 100%; width: 100%;",
            ):
                # Wind tiles (composited with terrain)
                leaflet.LTileLayer(
                    url=("wind_url", f"/tiles/{{z}}/{{x}}/{{y}}.png?v={CACHE_BUST}"),
                    opacity=1.0,
                    minZoom=2,
                    maxZoom=5,
                )
                # Border tiles (transparent with black lines)
                leaflet.LTileLayer(
                    url=("border_url", f"/borders/{{z}}/{{x}}/{{y}}.png?v={CACHE_BUST}"),
                    opacity=1.0,
                    minZoom=2,
                    maxZoom=5,
                )

if __name__ == "__main__":
    server.start()