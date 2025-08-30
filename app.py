from __future__ import annotations
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# =============================
# SAYFA AYARLARI
# =============================
st.set_page_config(page_title="SF Crime Risk — Devriye Planlama (Prototip)", layout="wide")
st.title("SF Crime Risk — Devriye Planlama (Prototip)")

# =============================
# SABİTLER
# =============================
SF_TZ_OFFSET = -7  # PDT kabaca; prod'da pytz kullanın
LON_MIN, LON_MAX = -122.52, -122.36
LAT_MIN, LAT_MAX = 37.70, 37.82
GRID_STEPS = 14
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]
RISK_BANDS = {"Yüksek": (0.90, 1.00), "Orta": (0.70, 0.90), "Hafif": (0.50, 0.70)}

rng = np.random.default_rng(42)

# =============================
# YARDIMCI FONKSİYONLAR
# =============================

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")


def linspace(a: float, b: float, n: int) -> List[float]:
    return [a + (b - a) * i / n for i in range(n + 1)]


def build_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    lons = linspace(LON_MIN, LON_MAX, GRID_STEPS)
    lats = linspace(LAT_MIN, LAT_MAX, GRID_STEPS)
    features, rows = [], []
    idx = 0
    for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
            lon0, lon1 = lons[i], lons[i + 1]
            lat0, lat1 = lats[j], lats[j + 1]
            cell_id = f"cell_{idx:03d}"
            centroid_lon = (lon0 + lon1) / 2
            centroid_lat = (lat0 + lat1) / 2
            poly = [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]
            feat = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [poly]},
                "properties": {"id": cell_id, "i": i, "j": j, "centroid_lon": centroid_lon, "centroid_lat": centroid_lat},
            }
            features.append(feat)
            rows.append({"cell_id": cell_id, "i": i, "j": j, "centroid_lon": centroid_lon, "centroid_lat": centroid_lat})
            idx += 1
    return pd.DataFrame(rows), features

GRID_DF, GRID_FEATURES = build_grid()


def scenario_multipliers(scenario: Dict) -> float:
    mult = 1.0
    if scenario.get("hava") == "Kötümser":
        mult *= 1.10
    elif scenario.get("hava") == "İyimser":
        mult *= 0.95
    if scenario.get("etkinlik", True):
        mult *= 1.05
    if scenario.get("cagri_modeli") == "Yan Model":
        mult *= 1.03
    return mult


def base_spatial_intensity(lon: float, lat: float) -> float:
    peak1 = math.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = math.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    noise = 0.1 * rng.random()
    return 0.2 + 0.8 * (peak1 + peak2) + noise


def hourly_forecast(start: datetime, horizon_h: int, scenario: Dict) -> pd.DataFrame:
    mult = scenario_multipliers(scenario)
    hours = [start + timedelta(hours=h) for h in range(horizon_h)]
    recs = []
    for _, row in GRID_DF.iterrows():
        lon, lat = row["centroid_lon"], row["centroid_lat"]
        base = base_spatial_intensity(lon, lat)
        for ts in hours:
            hour = ts.hour
            diurnal = 1.0 + 0.4 * math.sin((hour - 18) / 24 * 2 * math.pi)
            p = np.clip(base * diurnal * mult, 0, 1)
            p_any = float(np.clip(0.05 + 0.5 * p, 0, 0.98))
            alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
            w = rng.dirichlet(alpha)
            types = {t: float(p_any * w[k]) for k, t in enumerate(CRIME_TYPES)}
            q10 = float(max(0.0, p_any - 0.08))
            q90 = float(min(1.0, p_any + 0.08))
            recs.append({"cell_id": row["cell_id"], "ts": ts.isoformat(), "p_any": p_any, "q10": q10, "q90": q90, **types})
    return pd.DataFrame(recs)


def daily_forecast(start: datetime, days: int, scenario: Dict) -> pd.DataFrame:
    hourly = hourly_forecast(start, days * 24, scenario)
    hourly["date"] = hourly["ts"].str.slice(0, 10)
    daily = (
        hourly.groupby(["cell_id", "date"]).agg({"p_any": "mean", "q10": "mean", "q90": "mean", **{t: "mean" for t in CRIME_TYPES}}).reset_index()
    )
    return daily


def aggregate_for_view(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("cell_id").agg({"p_any": "mean", "q10": "mean", "q90": "mean", **{t: "mean" for t in CRIME_TYPES}}).reset_index()


def top_risky_table(df_agg: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    cols = ["cell_id", "p_any"] + CRIME_TYPES
    tab = df_agg[cols].sort_values("p_any", ascending=False).head(n).reset_index(drop=True)
    tab["p_any"] = tab["p_any"].round(3)
    for t in CRIME_TYPES:
        tab[t] = tab[t].round(3)
    return tab


def percentile_threshold(series: pd.Series, p: float) -> float:
    return float(np.quantile(series.to_numpy(), p))


def choose_candidates(df_agg: pd.DataFrame, bands: List[str]) -> pd.DataFrame:
    lo, hi = 1.0, 0.0
    for b in bands:
        bl, bh = RISK_BANDS[b]
        lo = min(lo, bl)
        hi = max(hi, bh)
    thr_lo = percentile_threshold(df_agg["p_any"], lo)
    thr_hi = percentile_threshold(df_agg["p_any"], hi)
    cand = df_agg[(df_agg["p_any"] >= thr_lo) & (df_agg["p_any"] <= thr_hi)].copy()
    return cand.sort_values("p_any", ascending=False)


def kmeans_like(coords: np.ndarray, weights: np.ndarray, k: int, iters: int = 20):
    n = len(coords)
    k = min(k, n)
    idx_sorted = np.argsort(-weights)
    centroids = coords[idx_sorted[:k]].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        for c in range(k):
            m = assign == c
            if not np.any(m):
                centroids[c] = coords[idx_sorted[min(0, n-1)]]
            else:
                w = weights[m][:, None]
                centroids[c] = (coords[m] * w).sum(axis=0) / max(1e-6, w.sum())
    return centroids, assign


def allocate_patrols(df_agg: pd.DataFrame, k: int, bands: List[str]) -> Dict:
    cand = choose_candidates(df_agg, bands)
    if cand.empty:
        return {"zones": []}
    merged = cand.merge(GRID_DF, on="cell_id")
    coords = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["p_any"].to_numpy()
    k = min(k, max(1, len(merged) // 3))
    cents, assign = kmeans_like(coords, weights, k)
    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy()
        cz = cents[z]
        angles = np.arctan2(sub["centroid_lat"] - cz[1], sub["centroid_lon"] - cz[0])
        sub = sub.assign(angle=angles).sort_values("angle")
        route = sub[["centroid_lat", "centroid_lon"]].to_numpy().tolist()
        zones.append({"id": f"Z{z+1}", "centroid": {"lat": float(cz[1]), "lon": float(cz[0])}, "cells": sub["cell_id"].tolist(), "route": route, "expected_risk": float(sub["p_any"].mean())})
    return {"zones": zones}


def color_for_percentile(p: float) -> str:
    if p >= 0.9: return "#b30000"
    if p >= 0.7: return "#ff7f00"
    if p >= 0.5: return "#ffff33"
    if p >= 0.3: return "#a1d99b"
    return "#c7e9c0"


def build_map(df_agg: pd.DataFrame, patrol: Dict | None = None) -> folium.Map:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    values = df_agg["p_any"].to_numpy()
    for feat in GRID_FEATURES:
        cid = feat["properties"]["id"]
        row = df_agg[df_agg["cell_id"] == cid]
        if row.empty:
            continue
        p = float(row["p_any"].iloc[0])
        q10 = float(row["q10"].iloc[0])
        q90 = float(row["q90"].iloc[0])
        types = {t: float(row[t].iloc[0]) for t in CRIME_TYPES}
        top3 = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
        top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])
        popup_html = f"""
        <b>{cid}</b><br/>
        p_any: {p:.2f} (q10={q10:.2f}, q90={q90:.2f})<br/>
        <b>En olası 3 tip</b>
        <ul style='margin-left:12px'>{top_html}</ul>
        <i>Seçili ufuk için ortalama risk</i>
        """
        style = {"fillColor": color_for_percentile(p), "color": "#666666", "weight": 0.5, "fillOpacity": 0.6}
        folium.GeoJson(data=feat, style_function=lambda x, s=style: s, tooltip=folium.Tooltip(f"{cid} — risk {p:.2f}"), popup=folium.Popup(popup_html, max_width=280)).add_to(m)

    thr99 = np.quantile(values, 0.99)
    urgent = df_agg[df_agg["p_any"] >= thr99]
    for _, r in urgent.iterrows():
        lat = GRID_DF.loc[GRID_DF["cell_id"] == r["cell_id"], "centroid_lat"].values[0]
        lon = GRID_DF.loc[GRID_DF["cell_id"] == r["cell_id"], "centroid_lon"].values[0]
        folium.CircleMarker(location=[lat, lon], radius=6, color="#000", fill=True, fill_color="#ff0000", popup=folium.Popup("ACİL — üst %1 risk", max_width=150)).add_to(m)

    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker([z["centroid"]["lat"], z["centroid"]["lon"]], icon=folium.DivIcon(html=f"<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'> {z['id']} </div>")).add_to(m)
    return m

# =============================
# UI — SİDEBAR
# =============================
st.sidebar.header("Ayarlar")
ufuk = st.sidebar.radio("Ufuk", options=["24s", "72s", "7g"], index=0, horizontal=True)
hava = st.sidebar.radio("Hava senaryosu", options=["Temel", "Kötümser", "İyimser"], index=0)
etkinlik = st.sidebar.checkbox("Etkinlik etkisi açık", value=True)
cagri = st.sidebar.radio("911/311 senaryosu", options=["Naif", "Yan Model"], index=0)

st.sidebar.header("Devriye Tahsisi")
K = st.sidebar.slider("Devriye sayısı (K)", 1, 20, 6, 1)
bands = st.sidebar.multiselect("Kapsanacak risk bandları", list(RISK_BANDS.keys()), default=["Yüksek", "Orta"])

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol = colB.button("Devriye öner")

# =============================
# STATE
# =============================
if "forecast" not in st.session_state:
    st.session_state["forecast"] = None
    st.session_state["agg"] = None
    st.session_state["patrol"] = None

# =============================
# ANA BÖLÜM
# =============================
col1, col2 = st.columns([2.4, 1.0])

with col1:
    st.caption(f"Son güncelleme (SF): {now_sf_iso()}")
    if btn_predict or st.session_state["agg"] is None:
        start = datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)
        scenario = {"hava": hava, "etkinlik": etkinlik, "cagri_modeli": cagri}
        if ufuk in ("24s", "72s"):
            horizon = 24 if ufuk == "24s" else 72
            df = hourly_forecast(start, horizon, scenario)
        else:
            df = daily_forecast(start, 7, scenario)
        agg = aggregate_for_view(df)
        st.session_state["forecast"] = df
        st.session_state["agg"] = agg
        st.session_state["patrol"] = None
    agg = st.session_state["agg"]

    if agg is not None:
        m = build_map(agg, st.session_state["patrol"])
        st_folium(m, width=None, height=620)

with col2:
    st.subheader("KPI")
    if st.session_state["agg"] is not None:
        a = st.session_state["agg"]
        kpi_expected = round(float(a["p_any"].sum()), 2)
        kpi_urgent = int((a["p_any"] >= np.quantile(a["p_any"], 0.99)).sum())
        kpi_cells = int(len(a))
        c1, c2, c3 = st.columns(3)
        c1.metric("Beklenen olay (toplam)", kpi_expected)
        c2.metric("Acil bölge (>%99)", kpi_urgent)
        c3.metric("Hücre sayısı", kpi_cells)

    st.subheader("En riskli bölgeler")
    if st.session_state["agg"] is not None:
        st.dataframe(top_risky_table(st.session_state["agg"]))

    st.subheader("Devriye özeti")
    if btn_patrol and st.session_state["agg"] is not None:
        patrol = allocate_patrols(st.session_state["agg"], K, bands or list(RISK_BANDS.keys()))
        st.session_state["patrol"] = patrol
    patrol = st.session_state["patrol"]
    if patrol:
        rows = [{"zone": z["id"], "cells": len(z["cells"]), "avg_risk": round(z["expected_risk"], 3)} for z in patrol.get("zones", [])]
        st.dataframe(pd.DataFrame(rows))

    st.subheader("Dışa aktar")
    if st.session_state["agg"] is not None:
        csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
        st.download_button("CSV indir", data=csv, file_name=f"risk_export_{int(time.time())}.csv", mime="text/csv")
