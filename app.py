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
SF_TZ_OFFSET = -7  # PDT kabaca; prod'da pytz kullanın
LON_MIN, LON_MAX = -122.52, -122.36
LAT_MIN, LAT_MAX = 37.70, 37.82
GRID_STEPS = 14
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]
RISK_BANDS = {"Yüksek": (0.90, 1.00), "Orta": (0.70, 0.90), "Hafif": (0.50, 0.70)}
KEY_COL = "geoid"
CACHE_VERSION = "geo-v1"

rng = np.random.default_rng(42)

# =============================
# YARDIMCI FONKSİYONLAR
# =============================

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")


def linspace(a: float, b: float, n: int) -> List[float]:
    return [a + (b - a) * i / n for i in range(n + 1)]


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
        return pd.DataFrame(columns=[key_field,"centroid_lon","centroid_lat"]), []
    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geoid = str(props.get(key_field, "")).strip()
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

def scenario_multipliers(_scenario: Dict) -> float:
    # Gerçek hava/etkinlik verisi pipeline’dan geldiği için kullanıcı senaryosu yok
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
    agg_map = {"p_any": "mean", "q10": "mean", "q90": "mean"}
    agg_map.update({t: "mean" for t in CRIME_TYPES})
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

def aggregate_for_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # her saat için lambda (beklenen olay sayısı)
    df["lam"] = p_to_lambda(df["p_any"])

    # saatlik belirsizliği ortalama tutalım (popup için)
    mean_map = {"q10": "mean", "q90": "mean"}

    # ufuk boyunca beklenen toplam (λ toplama kuralıyla)
    sum_map = {"lam": "sum"} | {t: "sum" for t in CRIME_TYPES}

    mean_part = df.groupby(KEY_COL, as_index=False).agg(mean_map)
    sum_part  = df.groupby(KEY_COL, as_index=False).agg(sum_map)

    out = mean_part.merge(sum_part, on=KEY_COL)
    out = out.rename(columns={"lam": "expected"})  # beklenen olay sayısı (λ)

    # Öncelik sınıfları: expected üzerinden quantile
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

    # Poisson’tan olasılıklar (λ = expected)
    lam = tab["expected"].to_numpy()
    tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]
    tab["P(≥2)%"] = [round(prob_ge_k(l, 2) * 100, 1) for l in lam]
    tab["P(≥3)%"] = [round(prob_ge_k(l, 3) * 100, 1) for l in lam]

    tab["expected"] = tab["expected"].round(2)
    for t in CRIME_TYPES:
        tab[t] = tab[t].round(3)

    return tab.rename(columns={"expected": "E[olay] (λ)"})

# --- Poisson yardımcıları (p_any -> lambda ve olasılıklar) ---
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

def percentile_threshold(series: pd.Series, p: float) -> float:
    return float(np.quantile(series.to_numpy(), p))

def choose_candidates(df_agg: pd.DataFrame, bands: List[str]) -> pd.DataFrame:
    # bands = ["Yüksek", "Orta"] vs; percentiller RISK_BANDS sözlüğünden
    lo, hi = 1.0, 0.0
    for b in bands:
        bl, bh = RISK_BANDS[b]
        lo = min(lo, bl)
        hi = max(hi, bh)
    # Percentiller expected üzerinden
    thr_lo = np.quantile(df_agg["expected"].to_numpy(), lo)
    thr_hi = np.quantile(df_agg["expected"].to_numpy(), hi)
    cand = df_agg[(df_agg["expected"] >= thr_lo) & (df_agg["expected"] <= thr_hi)].copy()
    return cand.sort_values("expected", ascending=False)

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

def allocate_patrols(df_agg: pd.DataFrame, bands: List[str]) -> Dict:
    """
    K'yi kullanıcıdan almak yerine otomatik seçer.
    Heuristik: k ≈ ceil(sqrt(n/2)), 1..20 aralığına sıkıştırılır.
    """
    cand = choose_candidates(df_agg, bands)
    if cand.empty:
        return {"zones": []}

    merged  = cand.merge(GEO_DF, on=KEY_COL)
    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy()

    n = len(merged)
    k_auto = int(np.ceil(np.sqrt(max(1, n) / 2)))
    k_auto = max(1, min(20, k_auto))

    cents, assign = kmeans_like(coords, weights, k_auto)

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
        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz[1]), "lon": float(cz[0])},
            "cells": sub[KEY_COL].astype(str).tolist(),
            "route": route,
            "expected_risk": float(sub["expected"].mean()),
        })
    return {"zones": zones}

def color_for_tier(tier: str) -> str:
    return {"Yüksek": "#d62728", "Orta": "#ff7f0e", "Hafif": "#1f77b4"}.get(tier, "#1f77b4")

def color_for_percentile(p: float) -> str:
    if p >= 0.9: return "#b30000"
    if p >= 0.7: return "#ff7f00"
    if p >= 0.5: return "#ffff33"
    if p >= 0.3: return "#a1d99b"
    return "#c7e9c0"

def build_map(df_agg: pd.DataFrame, patrol: Dict | None = None) -> folium.Map:
    # Boş/eksik durumda güvenli çık
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty or KEY_COL not in df_agg.columns:
        return m

    # Boyama ve "acil" eşiği expected üzerinden
    values = df_agg["expected"].to_numpy()

    for feat in GEO_FEATURES:
        gid = feat["properties"]["id"]  # = geoid
        row = df_agg.loc[df_agg[KEY_COL] == gid]
        if row.empty:
            continue

        expected = float(row["expected"].iloc[0])
        tier     = str(row["tier"].iloc[0])
        q10      = float(row["q10"].iloc[0])
        q90      = float(row["q90"].iloc[0])
        types    = {t: float(row[t].iloc[0]) for t in CRIME_TYPES}

        top3 = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
        top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])

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

        folium.GeoJson(
            data=feat,
            style_function=lambda _x, s=style: s,
            tooltip=folium.Tooltip(f"{gid} — E[olay]: {expected:.2f} — {tier}"),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    # En yüksek %1 beklenen olaya kırmızı uyarı
    thr99 = np.quantile(values, 0.99)
    urgent = df_agg[df_agg["expected"] >= thr99]
    for _, r in urgent.iterrows():
        lat = GEO_DF.loc[GEO_DF[KEY_COL] == r[KEY_COL], "centroid_lat"].values[0]
        lon = GEO_DF.loc[GEO_DF[KEY_COL] == r[KEY_COL], "centroid_lon"].values[0]
        folium.CircleMarker(
            location=[lat, lon],
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
                    html=f"<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'> {z['id']} </div>"
                ),
            ).add_to(m)

    return m
    
def _mark_auto_patrol():
    st.session_state["auto_patrol"] = True
    
# =============================
# UI — SİDEBAR
# =============================
st.sidebar.header("Ayarlar")
ufuk = st.sidebar.radio("Ufuk", options=["24s", "48s", "7g"], index=0, horizontal=True)

# yeni: kullanıcı "şu andan + X saat sonra başlat"
start_offset = st.sidebar.slider("Başlangıç (şimdiden + saat)", 0, 72, 0)

st.sidebar.header("Devriye Tahsisi")
bands = st.sidebar.multiselect(
    "Kapsanacak risk bandları (devriye için)",
    list(RISK_BANDS.keys()),
    default=["Yüksek", "Orta"],
    key="bands_select"
)

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner", disabled=st.session_state.get("agg") is None)

st.sidebar.caption("• Tahmin et: risk haritasını günceller.  • Devriye öner: seçili bandlardaki bölgelerden otomatik küme ve rota üretir.")

st.sidebar.caption("• Tahmin et: risk haritasını günceller.  • Devriye öner: K kümeye bölüp rotaları çıkarır.")

# =============================
# STATE
# =============================
if "forecast" not in st.session_state:
    st.session_state["forecast"] = None
    st.session_state["agg"] = None
    st.session_state["patrol"] = None
    st.session_state["auto_patrol"] = False  

# =============================
# ANA BÖLÜM
# =============================
col1, col2 = st.columns([2.4, 1.0])

with col1:
    st.caption(f"Son güncelleme (SF): {now_sf_iso()}")
    if btn_predict or st.session_state["agg"] is None:
        start = (
            datetime.utcnow()
            + timedelta(hours=SF_TZ_OFFSET)    
            + timedelta(hours=start_offset)      
        ).replace(minute=0, second=0, microsecond=0)
        scenario = {}
        
        if ufuk in ("24s", "48s"):
            horizon = 24 if ufuk == "24s" else 48
            df = hourly_forecast_cached(start.isoformat(), horizon, scenario)
        else:
            df = daily_forecast_cached(start.isoformat(), 7, scenario)
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
        kpi_expected = round(float(a["expected"].sum()), 2)
        high = int((a["tier"] == "Yüksek").sum())
        mid  = int((a["tier"] == "Orta").sum())
        low  = int((a["tier"] == "Hafif").sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beklenen olay (ufuk)", kpi_expected)
        c2.metric("Yüksek öncelik", high)
        c3.metric("Öncelik", mid)
        c4.metric("Düşük", low)
    
        with st.expander("Öncelik kümeleri – geoid listeleri"):
            cc1, cc2, cc3 = st.columns(3)
            cc1.write(", ".join(a.loc[a["tier"]=="Yüksek", KEY_COL].astype(str).tolist()) or "—")
            cc2.write(", ".join(a.loc[a["tier"]=="Orta",   KEY_COL].astype(str).tolist()) or "—")
            cc3.write(", ".join(a.loc[a["tier"]=="Hafif",  KEY_COL].astype(str).tolist()) or "—")
    
    st.subheader("En riskli bölgeler")
    st.dataframe(top_risky_table(st.session_state["agg"]))
    if st.session_state["agg"] is not None:
        st.dataframe(top_risky_table(st.session_state["agg"]))
    
    st.subheader("Devriye özeti")
    
    # Butona basınca veya K/bands değişince yeniden öner
    if st.session_state.get("agg") is not None and (btn_patrol or st.session_state.get("auto_patrol", False)):
        st.session_state["patrol"] = allocate_patrols(
            st.session_state["agg"], bands or list(RISK_BANDS.keys())
        )
        # bir kez çalışsın, bayrağı sıfırla
        st.session_state["auto_patrol"] = False
    
    patrol = st.session_state.get("patrol")
    if patrol:
        rows = [{"zone": z["id"], "cells": len(z["cells"]),
                 "avg_risk": round(z["expected_risk"], 3)} for z in patrol.get("zones", [])]
        st.dataframe(pd.DataFrame(rows))

    st.subheader("Dışa aktar")
    if st.session_state["agg"] is not None:
        csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
        st.download_button("CSV indir", data=csv, file_name=f"risk_export_{int(time.time())}.csv", mime="text/csv")
