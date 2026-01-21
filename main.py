from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import deckgl
import pydeck as pdk

# Server setup
server = get_server()
state = server.state
state.trame__title = "Wind Direction"

# Initial view - matching your Leaflet settings
initial_view = pdk.ViewState(
    latitude=20,
    longitude=0,
    zoom=3,
    min_zoom=1.5,
    max_zoom=10,
    pitch=0,  # Keep flat, no 3D tilt
    bearing=0,  # No rotation
)

# Create deck with Carto basemap (no API key needed)
deck = pdk.Deck(
    initial_view_state=initial_view,
    map_provider="carto",
    map_style="light",  # Clean style with labels
    layers=[],  # Empty for now, will add wind data later
)

# UI
with SinglePageLayout(server) as layout:
    layout.title.set_text("Wind Direction")
    
    with layout.content:
        deckgl.Deck(
            deck=deck,
            style="height: 100%; width: 100%;",
        )

if __name__ == "__main__":
    server.start()