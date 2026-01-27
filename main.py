"""
Wind visualization with pre-rendered tiles.
"""

import os
from aiohttp import web
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v3, leaflet3 as leaflet

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
        return web.FileResponse(tile_path)
    return web.Response(status=404)

# Register route after server is ready
@server.controller.add("on_server_bind")
def on_bind(wslink_server):
    wslink_server.app.router.add_get("/tiles/{z}/{x}/{y}.png", serve_tile)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Speed")
    
    with layout.content:
        with v3.VContainer(fluid=True, classes="fill-height pa-0"):
            with leaflet.LMap(
                zoom=("zoom", 2),
                center=("center", [20, 0]),
                world_copy_jump=True,
                max_zoom=5,
                min_zoom=3,
                zoom_snap=0,
                zoom_delta=0.25,
                wheel_px_per_zoom_level=200,
                style="height: 100%; width: 100%;",
            ):
                # Dark basemap
                leaflet.LTileLayer(
                    url="https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png",
                    minZoom=2,
                    maxZoom=5,
                )

                # Wind tiles
                leaflet.LTileLayer(
                    url=("wind_url", "/tiles/{z}/{x}/{y}.png"),
                    opacity=0.3,
                    minZoom=2,
                    maxZoom=5,
                )

if __name__ == "__main__":
    server.start()