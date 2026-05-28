import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import json
import rasterio
import numpy as np
from rasterio.warp import transform_bounds
import geopandas as gpd
import requests
from matplotlib.colors import ListedColormap
from rasterio.warp import reproject, Resampling, calculate_default_transform

# The 10 CLC+ Backbone classes — same set as defined in 1-acquisition.qmd § Exercise 10.
# Keep the order, IDs and hex colours in sync with that file (and with 5-statistics.qmd
# and map-nuts.qmd) so legends are consistent across the tutorial.
classes = [
    ("Sealed (1)",                        "#FF0100"),
    ("Woody – needle leaved trees (2)",   "#238B23"),
    ("Woody – broadleaved deciduous (3)", "#80FF00"),
    ("Woody – broadleaved evergreen (4)", "#00FF00"),
    ("Low-growing woody plants (5)",      "#804000"),
    ("Permanent herbaceous (6)",          "#CCF24E"),
    ("Periodically herbaceous (7)",       "#FEFF80"),
    ("Lichens and mosses (8)",            "#FF81FF"),
    ("Non- and sparsely-vegetated (9)",   "#BFBFBF"),
    ("Water (10)",                        "#0080FF"),
]

cmap = ListedColormap([color for _, color in classes])
label_to_color = {i + 1: color for i, (_, color) in enumerate(classes)}

api_url = "https://funathon-2026-project3-api.lab.sspcloud.fr"

# Titre
st.title("Ma carte Folium avec Streamlit")

# Coordonnées de départ (Paris)
latitude = 45.763497
longitude = 4.8369502

gps_point = [latitude, longitude]  # [lat, lon]
year = 2024
dst_crs = "EPSG:4326"

@st.cache_data
def find_image(gps_point, year):

    response_find = requests.get(
        f"{api_url}/find_image",
        params={
            "gps_point": gps_point,
            "year": year,
        },
    )

    return response_find.json()

image_filepath = find_image(gps_point, year)

tile_url = (
    f"https://minio.lab.sspcloud.fr/{image_filepath}"
)

@st.cache_data
def load_raster_data(tile_url, dst_crs):
    with rasterio.open(tile_url) as src:
        rgb_data = src.read([4, 3, 2])
        tile_crs = src.crs
        tile_bounds = src.bounds
        tile_transform = src.transform

    h, w = rgb_data.shape[1], rgb_data.shape[2]

    dst_transform, dst_w, dst_h = calculate_default_transform(
        tile_crs, dst_crs, w, h, *tile_bounds
    )

    rgb_wgs84_bands = np.zeros((3, dst_h, dst_w), dtype=np.float32)

    for i in range(3):
        reproject(
            source=rgb_data[i],
            destination=rgb_wgs84_bands[i],
            src_transform=tile_transform,
            src_crs=tile_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    p98 = np.percentile(rgb_wgs84_bands[rgb_wgs84_bands > 0], 98)

    rgb_wgs84 = np.clip(
        np.transpose(rgb_wgs84_bands, (1, 2, 0)) / p98,
        0,
        1,
    )

    alpha = (rgb_wgs84_bands.max(axis=0) > 0).astype(np.float32)
    rgba_wgs84 = np.dstack([rgb_wgs84, alpha])

    west, south, east, north = transform_bounds(
        tile_crs, dst_crs, *tile_bounds
    )

    return rgba_wgs84, west, south, east, north


@st.cache_data
def load_predictions(api_url, image_filepath, dst_crs):
    response_pred = requests.get(
        f"{api_url}/predict_image",
        params={"image": image_filepath, "polygons": True},
    )

    gdf_pred = gpd.GeoDataFrame.from_features(
        json.loads(response_pred.json())["features"],
        crs="EPSG:3035",
    )

    return gdf_pred.to_crs(dst_crs)

rgba_wgs84, west, south, east, north = load_raster_data(
    tile_url,
    dst_crs
)

gdf_pred = load_predictions(
    api_url,
    image_filepath,
    dst_crs
)

# Création de la carte
m = folium.Map(
    location=[latitude, longitude],
    zoom_start=12
)

fg_image = folium.FeatureGroup(name="Sentinel-2 RGB", show=True)
ImageOverlay(
    image=rgba_wgs84,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(fg_image)
fg_image.add_to(m)

fg_pred = folium.FeatureGroup(name="Predicted polygons", show=True)

folium.GeoJson(
    gdf_pred,
    style_function=lambda feature: {
        "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(fg_pred)
fg_pred.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
# Affichage dans Streamlit
st_folium(
    m,
    width=700,
    height=500,
    returned_objects=[]
)