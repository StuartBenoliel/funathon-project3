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

nuts3_centroids = {
    "Austria (AT332)": (47.2692, 11.4041),
    "Belgium (BE100)": (50.8503, 4.3517),
    "Cyprus (CY000)": (35.1856, 33.3823),
    "France (FRJ27)": (45.7640, 4.8357),  # Lyon-like
    "Croatia (HR050)": (45.8150, 15.9819),
    "Luxembourg (LU000)": (49.6116, 6.1319)
}
# Coordonnées de départ (Lyon)
if "lat" not in st.session_state:
    st.session_state.lat = 45.763497  # Lyon
if "lon" not in st.session_state:
    st.session_state.lon = 4.8369502

if "pending_lat" not in st.session_state:
    st.session_state.pending_lat = st.session_state.lat
if "pending_lon" not in st.session_state:
    st.session_state.pending_lon = st.session_state.lon

if "selected_nuts3" not in st.session_state:
    st.session_state.selected_nuts3 = "France (FRJ27)"

latitude = st.session_state.lat
longitude = st.session_state.lon
gps_point = [st.session_state.lat, st.session_state.lon]

with st.sidebar:
    st.header("Paramètres carte")

    options = list(nuts3_centroids.keys())

    selected = st.selectbox(
        "Choisir un NUTS3",
        options,
        index=options.index(st.session_state.selected_nuts3)
    )

    # 👉 si changement de NUTS3
    if selected != st.session_state.selected_nuts3:
        st.session_state.selected_nuts3 = selected

        lat, lon = nuts3_centroids[selected]

        # mise à jour point actif
        st.session_state.lat = lat
        st.session_state.lon = lon

        # sync pending aussi
        st.session_state.pending_lat = lat
        st.session_state.pending_lon = lon

        st.rerun()

    # édition manuelle
    st.session_state.pending_lat = st.number_input(
        "Latitude",
        value=st.session_state.pending_lat,
        format="%.6f"
    )

    st.session_state.pending_lon = st.number_input(
        "Longitude",
        value=st.session_state.pending_lon,
        format="%.6f"
    )

    if st.button("Valider coordonnées"):
        st.session_state.lat = st.session_state.pending_lat
        st.session_state.lon = st.session_state.pending_lon
        st.rerun()


gps_point = [latitude, longitude]  # [lat, lon]
year_a = 2024
year_b = 2021
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

image_filepath_a = find_image(gps_point, year_a)
image_filepath_b = find_image(gps_point, year_b)

tile_url_a = (
    f"https://minio.lab.sspcloud.fr/{image_filepath_a}"
)

tile_url_b = (
    f"https://minio.lab.sspcloud.fr/{image_filepath_a}"
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
    tile_url_a,
    dst_crs
)

gdf_pred_a = load_predictions(
    api_url,
    image_filepath_a,
    dst_crs
)

gdf_pred_b = load_predictions(
    api_url,
    image_filepath_b,
    dst_crs
)


class_names = {
    1:  "Sealed",
    2:  "Woody – needle leaved",
    3:  "Woody – broadleaved deciduous",
    4:  "Woody – broadleaved evergreen",
    5:  "Low-growing woody plants",
    6:  "Permanent herbaceous",
    7:  "Periodically herbaceous",
    8:  "Lichens and mosses",
    9:  "Non- and sparsely-vegetated",
    10: "Water",
}

import pandas as pd
from great_tables import GT, style, loc
from streamlit.components.v1 import html

gdf_pred_b["area_m2"]    = gdf_pred_b.geometry.area
gdf_pred_b["area_km2"]   = gdf_pred_b["area_m2"] / 1e6
gdf_pred_b["class_name"] = gdf_pred_b["label"].map(class_names)


gdf_pred_a["area_m2"]    = gdf_pred_a.geometry.area
gdf_pred_a["area_km2"]   = gdf_pred_a["area_m2"] / 1e6
gdf_pred_a["class_name"] = gdf_pred_a["label"].map(class_names)

def compute_stats(gdf):
    stats = (
        gdf.groupby(["label", "class_name"])
        .agg(
            n_polygons           = ("geometry", "count"),
            total_area_km2       = ("area_km2", "sum"),
            mean_polygon_area_m2 = ("area_m2",  "mean"),
            max_polygon_area_m2  = ("area_m2",  "max"),
        )
        .reset_index()
        .sort_values("total_area_km2", ascending=False)
    )
    total = stats["total_area_km2"].sum()
    stats["share_pct"] = (stats["total_area_km2"] / total * 100).round(2)
    return stats, total


stats_tile_2021, _ = compute_stats(gdf_pred_b)
stats_tile_2024, _ = compute_stats(gdf_pred_a)

tile_2021 = stats_tile_2021.set_index("class_name")["share_pct"].rename("2021 (%)")
tile_2024 = stats_tile_2024.set_index("class_name")["share_pct"].rename("2024 (%)")

tile_comparison = pd.concat([tile_2021, tile_2024], axis=1).fillna(0)
tile_comparison["change (pp)"] = (tile_comparison["2024 (%)"] - tile_comparison["2021 (%)"]).round(2)

gt_table = (
    GT(tile_comparison.reset_index(), rowname_col="class_name")
    .tab_header(title="Land-cover share — single tile, 2021 → 2024")
    .fmt_number(decimals=1)
    .data_color(
        columns=["change (pp)"],
        palette=["red", "white", "green"],
        domain=[-5, 5],
    )
    .data_color(
        columns=["2021 (%)", "2024 (%)"],
        palette=["white", "steelblue"]
    )
)

center_lat = (south + north) / 2 
center_lon = (west + east) / 2 
map_center = [center_lat, center_lon] 

map_zoom = 13

# Création de la carte
m = folium.Map(
    location=map_center,
    zoom_start=map_zoom,
    control_scale=True
)

fg_image = folium.FeatureGroup(name="Sentinel-2 RGB", show=True)
ImageOverlay(
    image=rgba_wgs84,
    bounds=[[south, west], [north, east]],
    opacity=0.7,
).add_to(fg_image)
fg_image.add_to(m)

fg_pred = folium.FeatureGroup(name="CLC+ polygons 2024", show=True)

folium.GeoJson(
    gdf_pred_a,
    style_function=lambda feature: {
        "fillColor": label_to_color.get(feature["properties"]["label"], "#808080"),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.6,
    },
    tooltip=folium.GeoJsonTooltip(fields=["label"], aliases=["Class:"]),
).add_to(fg_pred)
fg_pred.add_to(m)

fg_pred = folium.FeatureGroup(name="CLC+ polygons 2021", show=True)

folium.GeoJson(
    gdf_pred_b,
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
map_data = st_folium(
    m,
    height=400,  # ~80vh
    returned_objects=["last_clicked", "center", "zoom"]
)


# --- 2. clic carte (coordonnées métier) ---
last_clicked = map_data.get("last_clicked") if map_data else None

if last_clicked:

    st.session_state.pending_lat = last_clicked["lat"]
    st.session_state.pending_lon = last_clicked["lng"]

    st.sidebar.info("Coordonnées mises à jour depuis la carte (non validées)")


legend_html = """
<div style="
    border:2px solid black;
    border-radius:10px;
    padding:15px;
    margin-top:10px;
    background-color:white;
">
    <h4 style="margin-top:0;">CLC+ Backbone classes</h4>

    <div style="
        display:grid;
        grid-template-columns: repeat(2, minmax(250px, 1fr));
        gap:8px;
    ">
"""

for label, color in classes:
    legend_html += f"""
    <div style="display:flex; align-items:center;">
        <div style="
            width:18px;
            height:18px;
            background:{color};
            border:1px solid black;
            margin-right:10px;
            flex-shrink:0;
        "></div>

        <span style="font-size:14px;">
            {label}
        </span>
    </div>
    """

legend_html += """
    </div>
</div>
"""

from streamlit.components.v1 import html

html(legend_html, height=250)

st.subheader("Land-cover statistics")

html(
    gt_table.as_raw_html(),
    height=500,
    scrolling=True
)