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
import json
from pathlib import Path

# =============================
# SAYFA AYARLARI
# =============================
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.title("SUTAM: Suç Tahmin Modeli")

# =============================
# SABİTLER
# =============================
SF_TZ_OFFSET = -7  # PDT kabaca; prod'da pytz/pytzdata kullanın
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]
KEY_COL = "geoid"
CACHE_VERSION = "v2-geo-poisson"

rng = np.random.default_rng(42)

# =============================
# YARDIMCI FONKSİYONLAR
# =============================

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

def polygon_centroid(lonlat_loop):
    # lonlat_loop: [[lon,lat], ..., ilk noktaya kapanan halka]
    x, y = zip(*lonlat_loop)
    A = Cx = Cy = 0.0
    for i in range(len(lonlat_loop) - 1):
        cross = x[i]*y[i+1] - x[i+1]*y[i]
        A  += cross
        Cx += (x[i] + x[i+1]) * cross
        Cy += (y[i] + y[i+1]) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return float(sum(x)/len(x)), float(sum(y)/len(y))
    return float(Cx/(6*A)), float(Cy/(6*A))

def load_geoid_layer(path="data/sf_cells.geojson", key_field=KEY_COL):
    p = Path(path)
    if not p.exists():
        st.error(f"GEOJSON bulunamadı: {path}")
        return pd.DataFrame(columns=[key_field, "centroid_lon", "centroid_lat"]), []
    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geoid = str(
            props.get(key_field)
            or props.get(key_field.upper())
            or props.get("GEOID")
            or props.get("geoid")
            or ""
        ).strip()
        if not geoid:
            continue
        lon = props.get("centroid_lon"); lat = props.get("centroid_lat")
        if lon is None or lat is None:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
            elif geom.get("type") == "MultiPolygon":
                ring = geom["coordinates"][0][0]
            else:
                continue
            lon, lat = polygon_centroid(ring)
        rows.append({key_field: geoid, "centroid_lon": float(lon), "centroid_lat": float(lat)})
        feat.setdefault("properties", {})["id"] = geoid  # tooltip/popup için hızlı id
        feats_out.append(feat)
    return pd.DataFrame(rows), feats_out

@st.cache_data
def load_geoid_layer_cached(path, key_field=KEY_COL):
    return load_geoid_layer(path, key_field)

GEO_DF, GEO_FEATURES = load_geoid_layer_cached("data/sf_cells.geojson", key_field=KEY_COL)
if GEO_DF.empty:
    st.error("GEOJSON yüklendi ama satır gelmedi. 'data/sf_cells.geojson' içinde 'properties.geoid' eksik olabilir.")

# --------- HIZLI AGGREGATION BLOĞU (EKLE: GEO_DF satırının hemen altına) ---------
@st.cache_data(show_spinner=False)
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()
    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    noise = 0.07  # küçük sabit gürültü
    return 0.2 + 0.8 * (peak1 + peak2) + noise

BASE_INT = precompute_base_intensity(GEO_DF)

def p_to_lambda_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 0.999999)
    return -np.log1p(-p)

@st.cache_data(show_spinner=False)
def aggregate_fast(start_iso: str, horizon_h: int) -> pd.DataFrame:
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    # tüm hücreler × saatler
    p = np.clip(BASE_INT[:, None] * diurnal[None, :], 0, 1)
    p_any = np.clip(0.05 + 0.5 * p, 0, 0.98)

    lam = p_to_lambda_array(p_any)
    expected = lam.sum(axis=1)
    q10 = np.maximum(0.0, p_any - 0.08).mean(axis=1)
    q90 = np.minimum(1.0, p_any + 0.08).mean(axis=1)

    alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
    W = rng.dirichlet(alpha, size=len(GEO_DF))
    types = expected[:, None] * W
    assault, burglary, theft, robbery, vandalism = types.T

    out = pd.DataFrame({
        KEY_COL: GEO_DF[KEY_COL].to_numpy(),
        "expected": expected,
        "q10": q10, "q90": q90,
        "assault": assault, "burglary": burglary, "theft": theft,
        "robbery": robbery, "vandalism": vandalism,
    })

    q90_thr = out["expected"].quantile(0.90)
    q70_thr = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90_thr, out["expected"] >= q70_thr],
        ["Yüksek", "Orta"], default="Hafif",
    )
    return out

# --- Sentetik yoğunluk (sadece prototip) ---
def scenario_multipliers(_scenario: Dict) -> float:
    return 1.0

def base_spatial_intensity(lon: float, lat: float) -> float:
    peak1 = math.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = math.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    noise = 0.1 * rng.random()
    return 0.2 + 0.8 * (peak1 + peak2) + noise

def hourly_forecast(start: datetime, horizon_h: int, scenario: Dict) -> pd.DataFrame:
    mult = scenario_multipliers(scenario)
    hours = [start + timedelta(hours=h) for h in range(horizon_h)]
    recs = []
    for _, row in GEO_DF.iterrows():
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
            recs.append({KEY_COL: row[KEY_COL], "ts": ts.isoformat(),
                         "p_any": p_any, "q10": q10, "q90": q90, **types})
    return pd.DataFrame(recs)

def daily_forecast(start: datetime, days: int, scenario: Dict) -> pd.DataFrame:
    hourly = hourly_forecast(start, days * 24, scenario)
    hourly["date"] = hourly["ts"].str.slice(0, 10)
    agg_map = {"p_any": "mean", "q10": "mean", "q90": "mean"} | {t: "mean" for t in CRIME_TYPES}
    daily = hourly.groupby([KEY_COL, "date"], as_index=False).agg(agg_map)
    return daily

@st.cache_data(show_spinner=False)
def hourly_forecast_cached(start_iso: str, horizon_h: int, scenario: Dict, _v=CACHE_VERSION):
    start = datetime.fromisoformat(start_iso)
    return hourly_forecast(start, horizon_h, scenario)

@st.cache_data(show_spinner=False)
def daily_forecast_cached(start_iso: str, days: int, scenario: Dict, _v=CACHE_VERSION):
    start = datetime.fromisoformat(start_iso)
    return daily_forecast(start, days, scenario)

# --- Poisson yardımcıları ---
def p_to_lambda(p: pd.Series | np.ndarray) -> np.ndarray:
    # p_any = 1 - e^{-λ}  =>  λ = -ln(1 - p)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 0.999999)
    return -np.log(1.0 - p)

def pois_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(k + 1):
        s += (lam ** i) / math.factorial(i)
    return math.exp(-lam) * s

def prob_ge_k(lam: float, k: int) -> float:
    # P(N >= k) = 1 - CDF(k-1; λ)
    return 1.0 - pois_cdf(k - 1, lam)

# --- Görünüm için agregasyon ---
def aggregate_for_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # saatlik/periodik p_any → λ
    df["lam"] = p_to_lambda(df["p_any"])

    # saatlik belirsizliklerin ortalaması (popup için)
    mean_part = df.groupby(KEY_COL, as_index=False).agg({"q10": "mean", "q90": "mean"})
    # ufuk boyunca beklenen toplam olay (λ toplamı)
    sum_part  = df.groupby(KEY_COL, as_index=False).agg({"lam": "sum"} | {t: "sum" for t in CRIME_TYPES})

    out = mean_part.merge(sum_part, on=KEY_COL).rename(columns={"lam": "expected"})

    # Öncelik sınıfları (quantile) – expected üzerinden
    q90 = out["expected"].quantile(0.90)
    q70 = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90, out["expected"] >= q70],
        ["Yüksek", "Orta"],
        default="Hafif",
    )
    return out

def top_risky_table(df_agg: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    tab = df_agg[[KEY_COL, "expected"] + CRIME_TYPES].sort_values(
        "expected", ascending=False
    ).head(n).reset_index(drop=True)

    lam = tab["expected"].to_numpy()
    tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]
    tab["P(≥2)%"] = [round(prob_ge_k(l, 2) * 100, 1) for l in lam]
    tab["P(≥3)%"] = [round(prob_ge_k(l, 3) * 100, 1) for l in lam]

    tab["expected"] = tab["expected"].round(2)
    for t in CRIME_TYPES:
        tab[t] = tab[t].round(3)

    return tab.rename(columns={"expected": "E[olay] (λ)"})

# --- Devriye kümeleme ---
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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# YENİ: K ve görev süresiyle devriye planlama
def allocate_patrols(
    df_agg: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
) -> Dict:
    """
    k_planned: planlanan ekip sayısı
    duty_minutes: her ekibin sahada geçireceği süre (dk)
    cell_minutes: 1 hücrede ort. kontrol süresi (dk)
    travel_overhead: % seyir/aktarma payı (0.40 = %40)
    """
    # Adaylar = Yüksek + Orta
    cand = df_agg[df_agg["tier"].isin(["Yüksek", "Orta"])].copy()
    if cand.empty:
        return {"zones": []}

    merged  = cand.merge(GEO_DF, on=KEY_COL)
    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy()

    k = max(1, min(int(k_planned), 50))
    cents, assign = kmeans_like(coords, weights, k)

    # Hücre kapasitesi (adet) ~ görev süresi / (hücre_süresi * (1+overhead))
    cap_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + travel_overhead))))

    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy().sort_values("expected", ascending=False)

        # kapasite kadar en riskli hücreyi al
        sub_planned = sub.head(cap_cells).copy()

        cz = cents[z]
        # rotayı açıya göre sırala
        angles = np.arctan2(sub_planned["centroid_lat"] - cz[1], sub_planned["centroid_lon"] - cz[0])
        sub_planned = sub_planned.assign(angle=angles).sort_values("angle")

        route = sub_planned[["centroid_lat", "centroid_lon"]].to_numpy().tolist()
        n_cells = len(sub_planned)
        eta_minutes = int(round(n_cells * cell_minutes * (1.0 + travel_overhead)))
        util = min(100, int(round(100 * eta_minutes / max(1, duty_minutes))))

        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz[1]), "lon": float(cz[0])},
            "cells": sub_planned[KEY_COL].astype(str).tolist(),
            "route": route,
            "expected_risk": float(sub_planned["expected"].mean()),
            "planned_cells": int(n_cells),
            "eta_minutes": int(eta_minutes),
            "utilization_pct": int(util),
            "capacity_cells": int(cap_cells),
        })
    return {"zones": zones}
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def color_for_tier(tier: str) -> str:
    return {"Yüksek": "#d62728", "Orta": "#ff7f0e", "Hafif": "#1f77b4"}.get(tier, "#1f77b4")

def build_map(
    df_agg: pd.DataFrame,
    patrol: Dict | None = None,
    show_popups: bool = True
) -> folium.Map:
    # Boş/eksik durumda güvenli çık
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty or KEY_COL not in df_agg.columns:
        return m

    # Boyama ve "acil" eşiği expected üzerinden
    values = df_agg["expected"].to_numpy()

    # Hızlı erişim için index
    idx = df_agg.set_index(KEY_COL)

    for feat in GEO_FEATURES:
        gid = feat["properties"].get("id")  # = geoid
        if gid not in idx.index:
            continue

        r = idx.loc[gid]
        expected = float(r["expected"])
        tier     = str(r["tier"])
        q10      = float(r["q10"])
        q90      = float(r["q90"])
        types    = [(t, float(r[t])) for t in CRIME_TYPES]

        # en olası 3 suç
        top3 = sorted(types, key=lambda x: x[1], reverse=True)[:3]
        top_html = "".join(f"<li>{t}: {v:.2f}</li>" for t, v in top3)

        popup_html = f"""
        <b>{gid}</b><br/>
        E[olay] (ufuk): {expected:.2f} &nbsp;•&nbsp; Öncelik: <b>{tier}</b><br/>
        <b>En olası 3 tip</b>
        <ul style='margin-left:12px'>{top_html}</ul>
        <i>Belirsizlik (saatlik ort.): q10={q10:.2f}, q90={q90:.2f}</i>
        """

        style = {
            "fillColor": color_for_tier(tier),
            "color": "#666666",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }

        # GeoJSON'u ekle, popup'ı (opsiyonel) sonradan bağla
        geo = folium.GeoJson(
            data=feat,
            style_function=lambda _x, s=style: s,
            highlight_function=lambda _x: {"weight": 1.5, "color": "#000000", "fillOpacity": 0.7},
            tooltip=folium.Tooltip(f"{gid} — E[olay]: {expected:.2f} — {tier}"),
        )
        # Popup: sadece Yüksek/Orta için ekle (istersen 'or True' yapıp hepsine aç)
        if show_popups and tier != "Hafif":
            folium.Popup(popup_html, max_width=280).add_to(geo)
        geo.add_to(m)

    # En yüksek %1 beklenen olaya kırmızı uyarı
    if len(values):
        thr99 = np.quantile(values, 0.99)
        urgent = df_agg[df_agg["expected"] >= thr99]
        merged = urgent.merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
        for _, rr in merged.iterrows():
            if pd.isna(rr.get("centroid_lat")) or pd.isna(rr.get("centroid_lon")):
                continue
            folium.CircleMarker(
                location=[float(rr["centroid_lat"]), float(rr["centroid_lon"])],
                radius=6,
                color="#000",
                fill=True,
                fill_color="#ff0000",
                popup=folium.Popup("ACİL — üst %1 E[olay]", max_width=150),
            ).add_to(m)

    # Devriye rotaları
    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(
                    html=(
                        "<div style='background:#111;color:#fff;padding:2px 6px;"
                        "border-radius:6px'> {}</div>".format(z["id"])
                    )
                ),
            ).add_to(m)

    return m
    
def build_map_fast(df_agg: pd.DataFrame, show_popups: bool = False, patrol: Dict | None = None) -> folium.Map:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty:
        return m

    # Görselleştirme renkleri
    color_map = {r[KEY_COL]: color_for_tier(r["tier"]) for _, r in df_agg.iterrows()}

    # Veri sözlüğü (popup için hızlı erişim)
    data_map = df_agg.set_index(KEY_COL).to_dict(orient="index")

    # GEOJSON’u popup içeriğiyle zenginleştir
    features = []
    for feat in GEO_FEATURES:
        # derin kopya (global objeyi kirletmeyelim)
        f = json.loads(json.dumps(feat))
        gid = f["properties"].get("id")

        row = data_map.get(gid)
        if row:
            expected = float(row["expected"])
            tier     = str(row["tier"])
            q10      = float(row["q10"])
            q90      = float(row["q90"])
            types    = {t: float(row[t]) for t in CRIME_TYPES}
            top3     = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
            top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])

            popup_html = (
                f"<b>{gid}</b><br/>"
                f"E[olay] (ufuk): {expected:.2f} &nbsp;•&nbsp; Öncelik: <b>{tier}</b><br/>"
                f"<b>En olası 3 tip</b>"
                f"<ul style='margin-left:12px'>{top_html}</ul>"
                f"<i>Belirsizlik (saatlik ort.): q10={q10:.2f}, q90={q90:.2f}</i>"
            )
            f["properties"]["popup_html"] = popup_html
            f["properties"]["expected"]   = round(expected, 2)
            f["properties"]["tier"]       = tier
        features.append(f)

    fc = {"type": "FeatureCollection", "features": features}

    def style_fn(feat):
        gid = feat["properties"].get("id")
        return {
            "fillColor": color_map.get(gid, "#9ecae1"),
            "color": "#666666",
            "weight": 0.3,
            "fillOpacity": 0.55,
        }

    # Tooltip + Popup (opsiyonel)
    tooltip = folium.GeoJsonTooltip(
        fields=["id", "tier", "expected"],
        aliases=["GEOID", "Öncelik", "E[olay]"],
        localize=True,
        sticky=False,
    ) if show_popups else None

    popup = folium.GeoJsonPopup(
        fields=["popup_html"],
        labels=False,
        parse_html=False,
        max_width=280,
    ) if show_popups else None

    gj = folium.GeoJson(fc, style_function=style_fn, tooltip=tooltip, popup=popup)
    gj.add_to(m)

    # En yüksek %1 beklenen olaya kırmızı uyarı
    thr99 = np.quantile(df_agg["expected"].to_numpy(), 0.99)
    urgent = df_agg[df_agg["expected"] >= thr99]
    merged = urgent.merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL)
    for _, r in merged.iterrows():
        folium.CircleMarker(
            location=[r["centroid_lat"], r["centroid_lon"]],
            radius=5, color="#000", fill=True, fill_color="#ff0000",
            popup=None if not show_popups else folium.Popup("ACİL — üst %1 E[olay]", max_width=150)
        ).add_to(m)

    # Devriye rotaları
    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(
                    html=f"<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'> {z['id']} </div>"
                ),
            ).add_to(m)

    return m

# =============================
# UI — SİDEBAR
# =============================
st.sidebar.header("Ayarlar")
ufuk = st.sidebar.radio("Ufuk", options=["24s", "48s", "7g"], index=0, horizontal=True)

# Aralık seçimi (başlangıç-bitiş)
if ufuk == "24s":
    max_h, step = 24, 1
elif ufuk == "48s":
    max_h, step = 48, 3
else:
    max_h, step = 7 * 24, 24

start_h, end_h = st.sidebar.slider(
    "Zaman aralığı (şimdiden + saat)",
    min_value=0, max_value=max_h, value=(0, max_h), step=step
)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)", min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

# Devriye butonu (K/band yok)
colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner", disabled=st.session_state.get("agg") is None)
show_popups = st.sidebar.checkbox("Hücre pop-up (Top 3 suç)", value=True)

st.sidebar.caption("• Tahmin et: seçtiğin aralık için riskleri hesaplar.  • Devriye öner: K ekip ve görev süresine göre kümeler/rota üretir.")

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

    if btn_predict or st.session_state.get("agg") is None:
        start_dt = (
            datetime.utcnow()
            + timedelta(hours=SF_TZ_OFFSET + start_h)
        ).replace(minute=0, second=0, microsecond=0)

        horizon_h = max(1, end_h - start_h)
        start_iso = start_dt.isoformat()

        # Hızlı agregasyon
        agg = aggregate_fast(start_iso, horizon_h)

        st.session_state["forecast"] = None   # eski df kullanılmıyor
        st.session_state["agg"] = agg
        st.session_state["patrol"] = None     # yeni tahmin → devriye sıfırlansın

    agg = st.session_state.get("agg")

    if agg is not None:
        show_popups = st.checkbox(
            "Hücre popup'larını (en olası 3 suç) göster",
            value=True,
            help="Kapatırsanız harita biraz daha hızlı çalışır."
        )

        m = build_map_fast(
            agg,
            show_popups=show_popups,
            patrol=st.session_state.get("patrol")
        )
        st_folium(m, width=None, height=620)
    else:
        st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")
    
with col2:
    st.subheader("KPI")
    if st.session_state["agg"] is not None:
        a = st.session_state["agg"]
        kpi_expected = round(float(a["expected"].sum()), 2)
        high = int((a["tier"] == "Yüksek").sum())
        mid  = int((a["tier"] == "Orta").sum())
        low  = int((a["tier"] == "Hafif").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beklenen olay (ufuk)", kpi_expected)
        c2.metric("Yüksek", high); c3.metric("Orta", mid); c4.metric("Düşük", low)

        with st.expander("Öncelik kümeleri — geoid listeleri"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.write(", ".join(a.loc[a["tier"]=="Yüksek", KEY_COL].astype(str).tolist()) or "—")
            cc2.write(", ".join(a.loc[a["tier"]=="Orta",   KEY_COL].astype(str).tolist()) or "—")
            cc3.write(", ".join(a.loc[a["tier"]=="Hafif",  KEY_COL].astype(str).tolist()) or "—")

    st.subheader("En riskli bölgeler")
    if st.session_state["agg"] is not None:
        st.dataframe(top_risky_table(st.session_state["agg"]))

    st.subheader("Devriye özeti")
    if st.session_state.get("agg") is not None and btn_patrol:
        st.session_state["patrol"] = allocate_patrols(
            st.session_state["agg"],
            k_planned=K_planned,
            duty_minutes=int(duty_minutes),
            cell_minutes=int(cell_minutes),
            travel_overhead=0.40,
        )

    patrol = st.session_state.get("patrol")
    if patrol and patrol.get("zones"):
        rows = [{
            "zone": z["id"],
            "cells_planned": z["planned_cells"],
            "capacity_cells": z["capacity_cells"],
            "eta_minutes": z["eta_minutes"],
            "utilization_%": z["utilization_pct"],
            "avg_risk(E[olay])": round(z["expected_risk"], 2),
        } for z in patrol["zones"]]
        st.dataframe(pd.DataFrame(rows))

    st.subheader("Dışa aktar")
    if st.session_state["agg"] is not None:
        csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
        st.download_button("CSV indir", data=csv,
                           file_name=f"risk_export_{int(time.time())}.csv",
                           mime="text/csv")
