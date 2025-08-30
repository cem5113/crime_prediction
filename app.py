from __future__ import annotations
import math
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gradio as gr
import folium

# =============================
# ZAMAN VE BAZ AYARLAR
# =============================
SF_TZ_OFFSET = -7  # PDT varsayımı; prod'da dateutil/pytz ile tz yapılmalı

# SF yaklaşık alan (basitleştirilmiş bbox)
LON_MIN, LON_MAX = -122.52, -122.36
LAT_MIN, LAT_MAX = 37.70, 37.82
GRID_STEPS = 14  # 14x14 ~ 196 hücre

CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

# Risk bandları yüzdelik aralıkları
RISK_BANDS = {
    "Yüksek": (0.90, 1.00),
    "Orta": (0.70, 0.90),
    "Hafif": (0.50, 0.70),
}

# =============================
# YARDIMCI — GRID OLUŞTURMA
# =============================

def linspace(a: float, b: float, n: int) -> List[float]:
    return [a + (b - a) * i / n for i in range(n + 1)]


def build_grid() -> Tuple[pd.DataFrame, List[Dict]]:
    """Basit lat/lon kare grid ve GeoJSON Feature listesi döndürür."""
    lons = linspace(LON_MIN, LON_MAX, GRID_STEPS)
    lats = linspace(LAT_MIN, LAT_MAX, GRID_STEPS)
    features = []
    rows = []
    idx = 0
    for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
            lon0, lon1 = lons[i], lons[i + 1]
            lat0, lat1 = lats[j], lats[j + 1]
            cell_id = f"cell_{idx:03d}"
            centroid_lon = (lon0 + lon1) / 2
            centroid_lat = (lat0 + lat1) / 2
            poly = [
                [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
            ]
            feat = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [poly]},
                "properties": {
                    "id": cell_id,
                    "i": i,
                    "j": j,
                    "centroid_lon": centroid_lon,
                    "centroid_lat": centroid_lat,
                },
            }
            features.append(feat)
            rows.append({
                "cell_id": cell_id,
                "i": i,
                "j": j,
                "centroid_lon": centroid_lon,
                "centroid_lat": centroid_lat,
            })
            idx += 1
    df = pd.DataFrame(rows)
    return df, features

GRID_DF, GRID_FEATURES = build_grid()

# =============================
# MOCK TAHMİN MOTORU (DEMO)
# =============================

rng = np.random.default_rng(42)


def now_sf_iso() -> str:
    # Demo için kaba saat bazlı ofset
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")


def scenario_multipliers(scenario: Dict) -> Dict:
    """Senaryoya göre risk çarpanları (sadeleştirilmiş)."""
    mult = 1.0
    if scenario.get("hava") == "Kötümser":
        mult *= 1.10
    elif scenario.get("hava") == "İyimser":
        mult *= 0.95
    if scenario.get("etkinlik", True):
        mult *= 1.05
    if scenario.get("cagri_modeli") == "Yan Model":
        mult *= 1.03
    return {"global": mult}


def base_spatial_intensity(lon: float, lat: float) -> float:
    """Şehrin belirli bölgelerini sıcak yapacak basit uzamsal alan.
    Golden Gate Park çevresi düşük, merkez ve SOMA biraz yüksek gibi davranır.
    """
    # SOMA/Union Square civarı için bir tepe (yaklaşık -122.41, 37.78)
    peak1 = math.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    # Mission civarı (yaklaşık -122.42, 37.76)
    peak2 = math.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    # Gürültü
    noise = 0.1 * rng.random()
    return 0.2 + 0.8 * (peak1 + peak2) + noise


def hourly_forecast(start: datetime, horizon_h: int, scenario: Dict) -> pd.DataFrame:
    """Her hücre ve saat için p_any ve tip dağılımları üretir (mock)."""
    mult = scenario_multipliers(scenario)["global"]
    hours = [start + timedelta(hours=h) for h in range(horizon_h)]
    recs = []
    for _, row in GRID_DF.iterrows():
        lon, lat = row["centroid_lon"], row["centroid_lat"]
        base = base_spatial_intensity(lon, lat)
        for ts in hours:
            # diurnal pattern: akşam artar
            hour = ts.hour
            diurnal = 1.0 + 0.4 * math.sin((hour - 18) / 24 * 2 * math.pi)
            p = np.clip(base * diurnal * mult, 0, 1)
            # kalibrasyon için yumuşatma
            p_any = float(np.clip(0.05 + 0.5 * p, 0, 0.98))
            # tip dağılımı (Dirichlet)
            alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
            w = rng.dirichlet(alpha)
            types = {t: float(p_any * w[k]) for k, t in enumerate(CRIME_TYPES)}
            # belirsizlik bantları (oyuncak)
            q10 = float(max(0.0, p_any - 0.08))
            q90 = float(min(1.0, p_any + 0.08))
            recs.append({
                "cell_id": row["cell_id"],
                "ts": ts.isoformat(),
                "p_any": p_any,
                "q10": q10,
                "q90": q90,
                **types,
            })
    df = pd.DataFrame(recs)
    return df


def daily_forecast(start: datetime, days: int, scenario: Dict) -> pd.DataFrame:
    """Günlük toplam risk (mock)."""
    hourly = hourly_forecast(start, days * 24, scenario)
    hourly["date"] = hourly["ts"].str.slice(0, 10)
    daily = (
        hourly.groupby(["cell_id", "date"])
        .agg({"p_any": "mean", "q10": "mean", "q90": "mean", **{t: "mean" for t in CRIME_TYPES}})
        .reset_index()
    )
    return daily

# =============================
# DEVRIYE TAHSiSi (BASIT)
# =============================

def percentile_threshold(series: pd.Series, p: float) -> float:
    return float(np.quantile(series.to_numpy(), p))


def choose_candidates(df_agg: pd.DataFrame, bands: List[str]) -> pd.DataFrame:
    lo = 1.0
    hi = 0.0
    for b in bands:
        bl, bh = RISK_BANDS[b]
        lo = min(lo, bl)
        hi = max(hi, bh)
    # yüzdelik eşikleri global dağılıma göre uygula
    thr_lo = percentile_threshold(df_agg["p_any"], lo)
    thr_hi = percentile_threshold(df_agg["p_any"], hi)
    cand = df_agg[(df_agg["p_any"] >= thr_lo) & (df_agg["p_any"] <= thr_hi)].copy()
    return cand.sort_values("p_any", ascending=False)


def kmeans_like(coords: np.ndarray, weights: np.ndarray, k: int, iters: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Basit k-mean benzeri: ağırlıklı merkezlerle tekrar (sklearn'siz)."""
    n = len(coords)
    k = min(k, n)
    # Kötü başlangıçları azaltmak için en ağır k noktayı al
    idx_sorted = np.argsort(-weights)
    centroids = coords[idx_sorted[:k]].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        # atama
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        # güncelle
        for c in range(k):
            m = assign == c
            if not np.any(m):
                # boş küme → ağır bir noktayı çek
                centroids[c] = coords[idx_sorted[np.random.randint(0, min(10, n))]]
            else:
                w = weights[m][:, None]
                centroids[c] = (coords[m] * w).sum(axis=0) / max(1e-6, w.sum())
    return centroids, assign


def allocate_patrols(df_agg: pd.DataFrame, k: int, bands: List[str]) -> Dict:
    cand = choose_candidates(df_agg, bands)
    if cand.empty:
        return {"zones": []}
    # Koordinatlar
    merged = cand.merge(GRID_DF, left_on="cell_id", right_on="cell_id")
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
        # rota: merkez etrafında açıya göre sırala
        cz = cents[z]
        angles = np.arctan2(sub["centroid_lat"] - cz[1], sub["centroid_lon"] - cz[0])
        sub = sub.assign(angle=angles).sort_values("angle")
        route = sub[["centroid_lat", "centroid_lon"]].to_numpy().tolist()
        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz[1]), "lon": float(cz[0])},
            "cells": sub["cell_id"].tolist(),
            "route": route,
            "expected_risk": float(sub["p_any"].mean()),
        })
    return {"zones": zones}

# =============================
# HARITA
# =============================


def color_for_percentile(p: float) -> str:
    # basit gradyan (yeşil→sarı→kırmızı)
    if p >= 0.9:
        return "#b30000"
    if p >= 0.7:
        return "#ff7f00"
    if p >= 0.5:
        return "#ffff33"
    if p >= 0.3:
        return "#a1d99b"
    return "#c7e9c0"


def build_map(df_agg: pd.DataFrame, patrol: Dict | None = None) -> str:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    # yüzdelikler için global referans
    values = df_agg["p_any"].to_numpy()
    for feat in GRID_FEATURES:
        cid = feat["properties"]["id"]
        row = df_agg[df_agg["cell_id"] == cid]
        if row.empty:
            continue
        p = float(row["p_any"].iloc[0])
        q10 = float(row["q10"].iloc[0])
        q90 = float(row["q90"].iloc[0])
        # en olası 3 tip
        types = {t: float(row[t].iloc[0]) for t in CRIME_TYPES}
        top3 = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
        top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])
        popup_html = f"""
        <b>{cid}</b><br/>
        p_any: {p:.2f} (q10={q10:.2f}, q90={q90:.2f})<br/>
        <b>En olası 3 tip</b>
        <ul style='margin-left:12px'>{top_html}</ul>
        <i>Önümüzdeki ufuk için ortalama risk</i>
        """
        style = {
            "fillColor": color_for_percentile(p),
            "color": "#666666",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }
        gj = folium.GeoJson(
            data=feat,
            style_function=lambda x, s=style: s,
            tooltip=folium.Tooltip(f"{cid} — risk {p:.2f}"),
            popup=folium.Popup(popup_html, max_width=280),
        )
        gj.add_to(m)

    # Acil rozet: en üst %1
    thr99 = np.quantile(values, 0.99)
    urgent = df_agg[df_agg["p_any"] >= thr99]
    for _, r in urgent.iterrows():
        folium.CircleMarker(
            location=[
                GRID_DF.loc[GRID_DF["cell_id"] == r["cell_id"], "centroid_lat"].values[0],
                GRID_DF.loc[GRID_DF["cell_id"] == r["cell_id"], "centroid_lon"].values[0],
            ],
            radius=6,
            color="#000",
            fill=True,
            fill_color="#ff0000",
            popup=folium.Popup("ACİL — üst %1 risk", max_width=150),
        ).add_to(m)

    # Devriye bölgeleri ve rotalar
    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            # rota çizgisi
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(html=f"<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'> {z['id']} </div>"),
            ).add_to(m)

    return m._repr_html_()

# =============================
# ÖZET & TABLOLAR
# =============================

def aggregate_for_view(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Harita için hücre bazında ortalama risk (seçilen ufuk penceresi üzerinden)."""
    if mode in ("24s", "72s"):
        grp = df.groupby("cell_id").agg({
            "p_any": "mean",
            "q10": "mean",
            "q90": "mean",
            **{t: "mean" for t in CRIME_TYPES},
        }).reset_index()
    else:  # 7g
        grp = df.groupby("cell_id").agg({
            "p_any": "mean",
            "q10": "mean",
            "q90": "mean",
            **{t: "mean" for t in CRIME_TYPES},
        }).reset_index()
    return grp


def top_risky_table(df_agg: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    cols = ["cell_id", "p_any"] + CRIME_TYPES
    tab = df_agg[cols].sort_values("p_any", ascending=False).head(n).reset_index(drop=True)
    tab["p_any"] = tab["p_any"].round(3)
    for t in CRIME_TYPES:
        tab[t] = tab[t].round(3)
    return tab

# =============================
# GRADIO UI CALLBACKS
# =============================

STATE = {
    "forecast": None,  # saatlik/günlük ham
    "agg": None,       # harita için özet
    "patrol": None,
}


def do_predict(ufuk: str, hava: str, etkinlik: bool, cagri_modeli: str):
    start = datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)
    scenario = {"hava": hava, "etkinlik": etkinlik, "cagri_modeli": cagri_modeli}
    if ufuk == "24s":
        df = hourly_forecast(start, 24, scenario)
    elif ufuk == "72s":
        df = hourly_forecast(start, 72, scenario)
    else:
        df = daily_forecast(start, 7, scenario)
    agg = aggregate_for_view(df, ufuk)
    STATE["forecast"], STATE["agg"], STATE["patrol"] = df, agg, None
    html = build_map(agg, None)
    kpi_expected = round(float(agg["p_any"].sum()), 2)
    kpi_urgent = int((agg["p_any"] >= np.quantile(agg["p_any"], 0.99)).sum())
    kpi_cells = int(len(agg))
    table = top_risky_table(agg)
    return html, table, f"{kpi_expected}", f"{kpi_urgent}", f"{kpi_cells}", f"{now_sf_iso()}"


def do_patrol(k: int, bands: List[str]):
    agg = STATE.get("agg")
    if agg is None or agg.empty:
        return gr.update(), gr.update()
    patrol = allocate_patrols(agg, k, bands or list(RISK_BANDS.keys()))
    STATE["patrol"] = patrol
    html = build_map(agg, patrol)
    # küçük özet
    zones = patrol.get("zones", [])
    rows = []
    for z in zones:
        rows.append({
            "zone": z["id"],
            "cells": len(z["cells"]),
            "avg_risk": round(z["expected_risk"], 3),
        })
    table = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["zone", "cells", "avg_risk"])
    return html, table


def export_csv():
    agg = STATE.get("agg")
    if agg is None or agg.empty:
        return None
    ts = int(time.time())
    path = f"risk_export_{ts}.csv"
    agg.to_csv(path, index=False)
    return path

# =============================
# GRADIO — TEK SAYFA DÜZEN
# =============================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"), css="""
#metrics span {font-weight:600}
.right-col {min-width: 340px}
.lead {font-weight:700; font-size: 18px}
""") as demo:
    gr.Markdown("""
    ### SF Crime Risk — Devriye Planlama (Prototip)
    Kullanıcı dostu tek sayfa arayüz — saatlik/günlük risk, devriye tahsisi ve acil bölgeler.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Ayarlar**")
            ufuk = gr.Radio(["24s", "72s", "7g"], value="24s", label="Ufuk")
            hava = gr.Radio(["Temel", "Kötümser", "İyimser"], value="Temel", label="Hava senaryosu")
            etkinlik = gr.Checkbox(value=True, label="Etkinlik etkisi açık")
            cagri = gr.Radio(["Naif", "Yan Model"], value="Naif", label="911/311 senaryosu")
            btn_predict = gr.Button("Tahmin et", variant="primary")

            gr.Markdown("**Devriye Tahsisi**")
            k = gr.Slider(1, 20, value=6, step=1, label="Devriye sayısı (K)")
            bands = gr.CheckboxGroup(list(RISK_BANDS.keys()), value=["Yüksek", "Orta"], label="Kapsanacak risk bandları")
            btn_patrol = gr.Button("Devriye öner")

            gr.Markdown("**Dışa Aktar**")
            btn_export = gr.Button("CSV indir")
            dl = gr.File(label="İndirilen CSV")
        with gr.Column(scale=3):
            last_update = gr.Textbox(label="Son güncelleme (SF)", value=now_sf_iso(), interactive=False)
            map_html = gr.HTML(value="<div>Harita için 'Tahmin et' tıklayın.</div>")
            with gr.Row(elem_id="metrics"):
                m1 = gr.Label("—", label="Beklenen olay (toplam)")
                m2 = gr.Label("—", label="Acil bölge adedi (>%99)")
                m3 = gr.Label("—", label="Hücre sayısı")
        with gr.Column(scale=2, elem_classes=["right-col"]):
            gr.Markdown("**En riskli bölgeler**")
            top_tbl = gr.Dataframe(headers=["cell_id", "p_any"] + CRIME_TYPES, value=pd.DataFrame(columns=["cell_id", "p_any"] + CRIME_TYPES), interactive=False, wrap=True, height=300)
            gr.Markdown("**Devriye özet**")
            patrol_tbl = gr.Dataframe(headers=["zone", "cells", "avg_risk"], value=pd.DataFrame(columns=["zone", "cells", "avg_risk"]), interactive=False, height=220)

    # Etkileşimler
    btn_predict.click(
        fn=do_predict,
        inputs=[ufuk, hava, etkinlik, cagri],
        outputs=[map_html, top_tbl, m1, m2, m3, last_update],
    )

    btn_patrol.click(
        fn=do_patrol,
        inputs=[k, bands],
        outputs=[map_html, patrol_tbl],
    )

    btn_export.click(fn=export_csv, inputs=None, outputs=dl)

if __name__ == "__main__":
    demo.launch()
