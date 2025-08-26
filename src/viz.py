## 6) src/viz.py

```python
from __future__ import annotations
import folium
import pandas as pd
from streamlit_folium import st_folium

# Harita çizimi: centroid marker + intensity

def draw_map(topk_df: pd.DataFrame, centroids: pd.DataFrame, popup_cols=None, height=520):
    if popup_cols is None:
        popup_cols = ["GEOID","hour_range","p_crime","priority_score","top3_crime_types"]
    if centroids is None or centroids.empty or topk_df.empty:
        return None
    M = topk_df.merge(centroids, on="GEOID", how="left").dropna(subset=["lat","lon"])
    if M.empty:
        return None
    lat0, lon0 = M["lat"].median(), M["lon"].median()
    fmap = folium.Map(location=[lat0, lon0], zoom_start=12, control_scale=True)
    for _, r in M.iterrows():
        val = float(r.get("priority_score", 0))/100.0
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=6 + 10*val,
            popup=folium.Popup("<br/>".join([f"<b>{c}</b>: {r[c]}" for c in popup_cols]), max_width=300),
            fill=True,
            fill_opacity=0.7,
        ).add_to(fmap)
    return fmap
