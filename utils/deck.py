# utils/deck.py
from __future__ import annotations
import pydeck as pdk
import pandas as pd

try:
    from utils.constants import KEY_COL
except Exception:
    KEY_COL = "GEOID"

# Basit, hızlı pydeck haritası
def build_map_fast_deck(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    show_poi: bool = False,
    show_transit: bool = False,
    patrol: dict | None = None,
    show_hotspot: bool = True,
    show_temp_hotspot: bool = True,
    temp_hotspot_points: pd.DataFrame | None = None,
):
    center = [37.7749, -122.4194]
    layers = []

    # Hücre centroid’leri + renk (tier)
    if isinstance(df_agg, pd.DataFrame) and not df_agg.empty:
        cells = (
            df_agg.merge(geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
                  .dropna(subset=["centroid_lat", "centroid_lon"])
                  .copy()
        )
        color_map = {"Yüksek": [214, 39, 40], "Orta": [255, 127, 14], "Hafif": [31, 119, 180]}
        cells["color"] = cells["tier"].apply(lambda t: color_map.get(t, [31, 119, 180]))

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=cells,
            get_position="[centroid_lon, centroid_lat]",
            get_fill_color="color",
            get_radius=60,
            pickable=True,
            stroked=False,
            opacity=0.35,
        ))

        # Kalıcı hotspot: centroid ağırlıklı ısı
        if show_hotspot:
            hm = cells.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon", "expected": "weight"})
            hm["weight"] = hm["weight"].clip(lower=0)
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=hm[["lon", "lat", "weight"]],
                get_position="[lon, lat]",
                get_weight="weight",
                radius_pixels=60,
            ))

    # Geçici hotspot: olay noktaları ısı
    if show_temp_hotspot and isinstance(temp_hotspot_points, pd.DataFrame) and not temp_hotspot_points.empty:
        tmp = temp_hotspot_points.rename(columns={"latitude": "lat", "longitude": "lon"})
        if "weight" not in tmp.columns:
            tmp["weight"] = 1.0
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=tmp[["lon", "lat", "weight"]],
            get_position="[lon, lat]",
            get_weight="weight",
            radius_pixels=40,
        ))

    view_state = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=11.5)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"text": f"{KEY_COL}: {{{KEY_COL}}}\nE[olay]: {{expected}}"},
    )
    return deck
