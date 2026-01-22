import json
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import plotly
import plotly.express as px # type: ignore

# Load wind data
with open('data/wind_data.json') as f:
    wind_data = json.load(f)

print(f"Loaded {len(wind_data):,} wind points")

# Create figure
fig = px.scatter_map(
    wind_data, 
    lat='lat', 
    lon='lon', 
    color='speed',
    color_continuous_scale='Turbo', 
    zoom=1,
    center={'lat': 20, 'lon': 0}, 
    opacity=0.7
)
fig.update_traces(marker=dict(size=4))
fig.update_layout(
    mapbox_style='carto-darkmatter', 
    margin=dict(l=0, r=0, t=0, b=0)
)

# Trame setup
server = get_server()
state = server.state
state.trame__title = 'Wind Speed'

with SinglePageLayout(server) as layout:
    layout.title.set_text('Wind Speed')
    with layout.content:
        plotly.Figure(figure=fig, style='height: 100%; width: 100%;')

if __name__ == '__main__':
    server.start()