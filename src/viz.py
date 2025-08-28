# src/viz.py
from __future__ import annotations
import folium
import pandas as pd
from typing import Iterable, Optional

# Not: streamlit_folium yalnızca app tarafında import edilmeli (viz.py içinde gerek yok)

def _pick_color(val_01: float) -> str:
    """
    0-1 aralığındaki değeri kırmızı-sarı-yeşil renk geçişine çevir.
    Daha yüksek = daha riskli = daha kırmızı.
    """
    val = max(0.0, min(1.0, float(val_01)))
    # yeşil(0,170,0) -> sarı(255,200,0) -> kırmızı(220,20,60)
    if val < 0.5:
        t = val / 0.5
        r = int(0   * (1-t) + 255 * t)
        g = int(170 * (1-t) + 200 * t)
        b = int(0   * (1-t) +   0 * t)
    else:
        t = (val - 0.5) / 0.5
        r = int(255 * (1-t) + 220 * t)
        g = int(200 * (1-t) +  20 * t)
        b = int(0   * (1-t) +  60 * t)
    return f"#{r:02x}{g:02x}{b:02x}"

def _legend_html(title: str = "Öncelik Skoru (0→100)") -> str:
    # Basit gradient efsane
    gradient = (
        "linear-gradient(to right,"
        "#00aa00 0%, #8cc800 25%, #ffc800 50%, #ff7a30 75%, #dc143c 100%)"
    )
    return f"""
    <div style="
      position: fixed; bottom: 20px; left: 20px; z-index: 9999;
      background: rgba(255,255,255,0.9); padding: 10px 12px; border-radius: 8px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15); font-size: 12px;
    ">
      <div style="font-weight:600; margin-bottom:6px">{title}</div>
      <div style="width:220px; height:12px; background: {gradient}; border-radius: 6px;"></div>
      <div style="display:flex; justify-content:space-between; margin-top:4px;">
        <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
      </div>
    </div>
    """

def draw_map(
    topk_df: pd.DataFrame,
    centroids: pd.DataFrame,
    popup_cols: Optional[Iterable[str]] = None,
    *,
    use_cluster: bool = False,
    show_heatmap: bool = False,
    map_height: int = 560,
    zoom_start: int = 12
):
    """
    Top-K tablo + centroid'lerden etkileşimli harita üretir.

    Args
    ----
    topk_df : DataFrame
        En azından ['GEOID'] ve tercihen ['priority_score','p_crime','hour_range'] içermeli.
    centroids : DataFrame
        ['GEOID','lat','lon'] sütunları olmalı.
    popup_cols : Iterable[str] | None
        Popup'ta gösterilecek sütunlar (varsayılan: mantıklı bir set).
    use_cluster : bool
        Marker'ları MarkerCluster içine koy (çok nokta varsa faydalı).
    show_heatmap : bool
        Yoğunluk için HeatMap overlay ekle (priority_score ağırlıklı).
    map_height : int
        Streamlit'te gösterirken kullanılacak önerilen yükseklik (st_folium tarafında).
    zoom_start : int
        Başlangıç zoom'u.
    """
    if popup_cols is None:
        popup_cols = ["GEOID", "hour_range", "p_crime", "priority_score", "top3_crime_types"]

    if topk_df is None or topk_df.empty or centroids is None or centroids.empty:
        return None

    M = topk_df.merge(centroids, on="GEOID", how="left").dropna(subset=["lat", "lon"])
    if M.empty:
        return None

    # Görsel ölçüler ve renkleme
    # priority_score varsa onu, yoksa p_crime (0-1) * 100'ü kullan
    if "priority_score" in M.columns:
        prio01 = (M["priority_score"].astype(float).clip(0, 100)) / 100.0
    else:
        prio01 = M.get("p_crime", pd.Series(0.0, index=M.index)).astype(float).clip(0, 1)

    colors = prio01.apply(_pick_color)
    radii = (6 + 10 * prio01).astype(float)

    lat0, lon0 = M["lat"].median(), M["lon"].median()
    fmap = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, control_scale=True)

    # (opsiyonel) Heatmap
    if show_heatmap:
        try:
            from folium.plugins import HeatMap
            heat_data = M.assign(w=prio01.values)[["lat", "lon", "w"]].values.tolist()
            HeatMap(heat_data, radius=18, blur=22, max_zoom=15, min_opacity=0.2).add_to(fmap)
        except Exception:
            # HeatMap yoksa sessiz geç
            pass

    # (opsiyonel) Cluster
    group = fmap
    if use_cluster:
        try:
            from folium.plugins import MarkerCluster
            group = MarkerCluster().add_to(fmap)
        except Exception:
            group = fmap  # eklenti yoksa normal devam

    # Marker'lar
    for i, r in M.iterrows():
        val = float(prio01.loc[i])
        col = colors.loc[i]
        rad = float(radii.loc[i])

        # Popup HTML
        pops = []
        for c in popup_cols:
            if c in r:
                pops.append(f"<b>{c}</b>: {r[c]}")
        popup_html = "<br/>".join(pops) if pops else f"GEOID: {r['GEOID']}"

        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=rad,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.75,
            weight=1.0,
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(group)

    # Efsane
    fmap.get_root().html.add_child(folium.Element(_legend_html()))

    # Not: Streamlit'te göstermek için:
    # from streamlit_folium import st_folium
    # st_folium(fmap, height=map_height, width=None)

    return fmap
