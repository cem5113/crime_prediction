# app.py
from __future__ import annotations
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols
from utils.ui import (
    SMALL_UI_CSS, render_result_card, build_map_fast, render_kpi_row,
    render_day_hour_heatmap,  # <<< ısı matrisi
)
from utils.constants import (
    SF_TZ_OFFSET, KEY_COL,
    MODEL_VERSION, MODEL_LAST_TRAIN,
    CATEGORIES,
)
from components.last_update import show_last_update_badge

try:
    from utils.reports import load_events
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
    from reports import load_events  # utils/reports.py

# ── Sayfa ayarı: Streamlit'te en üstte olmalı
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)

# ── Başlık ve "Son güncelleme" rozetini göster
st.title("SUTAM: Suç Tahmin Modeli")

try:
    events_df = load_events("data/events.csv")
    if not events_df.empty and "ts" in events_df.columns:
        data_upto_val = pd.to_datetime(events_df["ts"]).max().date().isoformat()
    else:
        data_upto_val = None
except Exception:
    data_upto_val = None

show_last_update_badge(
    data_upto=data_upto_val,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

# ── Geo katmanı
GEO_DF, GEO_FEATURES = load_geoid_layer("data/sf_cells.geojson")
if GEO_DF.empty:
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.stop()

# ── Model tabanı
BASE_INT = precompute_base_intensity(GEO_DF)

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

# ── Sidebar
st.sidebar.markdown("### Görünüm")
sekme = st.sidebar.radio("", options=["Operasyon", "Raporlar"], index=0, horizontal=True)
st.sidebar.divider()

# ---- GÜNCELLENEN KISIM ----
st.sidebar.header("Devriye Parametreleri")

# Harita katmanları
st.sidebar.subheader("Harita katmanları")
show_poi = st.sidebar.checkbox("POI overlay", value=False)
show_transit = st.sidebar.checkbox("Toplu taşıma overlay", value=False)

# Ufuk seçimi
ufuk = st.sidebar.radio("Zaman Aralığı (şimdiden)", options=["24s", "48s", "7g"], index=0, horizontal=True)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7 * 24, 24)
start_h, end_h = st.sidebar.slider(
    "Saat filtresi",
    min_value=0, max_value=max_h, value=(0, max_h), step=step
)

# Kategori filtresi (Hepsi desteği ile)
sel_categories = st.sidebar.multiselect(
    "Kategori",
    ["(Hepsi)"] + CATEGORIES,
    default=[]
)
if sel_categories and "(Hepsi)" in sel_categories:
    filters = {"cats": CATEGORIES}   # tüm kategoriler
else:
    filters = {"cats": sel_categories or None}

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned    = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)",   min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner")
show_popups = st.sidebar.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)
# ---- GÜNCELLENEN KISIM ----

# ── State
if "agg" not in st.session_state:
    st.session_state.update({
        "agg": None, "patrol": None, "start_iso": None, "horizon_h": None, "explain": None
    })

# ── Operasyon
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")

        if btn_predict or st.session_state["agg"] is None:
            start_dt  = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
            horizon_h = max(1, end_h - start_h)
            start_iso = start_dt.isoformat()

            events_df = load_events("data/events.csv")  # ts, lat, lon kolonları olmalı

            # Tahmin (near-repeat parametreleri ile)
            agg = aggregate_fast(
                start_iso, horizon_h, GEO_DF, BASE_INT,
                events=events_df,
                near_repeat_alpha=0.35,
                nr_lookback_h=24,
                nr_radius_m=400,
                nr_decay_h=12.0,
                filters=filters,
            )

            st.session_state.update({
                "agg": agg, "patrol": None, "start_iso": start_iso, "horizon_h": horizon_h
            })

        agg = st.session_state["agg"]
        if agg is not None:
            m = build_map_fast(
                agg, GEO_FEATURES, GEO_DF,
                show_popups=show_popups,
                patrol=st.session_state.get("patrol"),
                show_poi=show_poi,            # <<< overlay bayrakları eklendi
                show_transit=show_transit,    # <<<
            )
            ret = st_folium(
                m, key="riskmap", height=540,
                returned_objects=["last_object_clicked", "last_clicked"]
            )
            if ret:
                gid, _ = resolve_clicked_gid(GEO_DF, ret)
                if gid:
                    st.session_state["explain"] = {"geoid": gid}

            start_iso  = st.session_state["start_iso"]
            horizon_h  = st.session_state["horizon_h"]
            info = st.session_state.get("explain")
            if info and info.get("geoid"):
                render_result_card(agg, info["geoid"], start_iso, horizon_h)
            else:
                st.info("Haritada bir hücreye tıklayın veya listeden seçin; kart burada görünecek.")
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

    with col2:
        st.subheader("Risk Özeti", anchor=False)

        if st.session_state["agg"] is not None:
            a = st.session_state["agg"]
            kpi_expected = round(float(a["expected"].sum()), 2)
            high = int((a["tier"] == "Yüksek").sum())
            mid  = int((a["tier"] == "Orta").sum())
            low  = int((a["tier"] == "Hafif").sum())

            render_kpi_row([
                ("Beklenen olay (ufuk)", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
                ("Yüksek",               high,         "Yüksek öncelikli hücre sayısı"),
                ("Orta",                 mid,          "Orta öncelikli hücre sayısı"),
                ("Düşük",                low,          "Düşük öncelikli hücre sayısı"),
            ])
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.subheader("En riskli bölgeler")
        if st.session_state["agg"] is not None:

            def top_risky_table(df_agg: pd.DataFrame, n: int = 12) -> pd.DataFrame:
                # --- CI95 ve Saat sütunları dahil
                def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
                    s = float(np.sqrt(max(lam, 1e-9)))
                    return max(0.0, lam - z * s), lam + z * s

                cols = [KEY_COL, "expected"]
                if "nr_boost" in df_agg.columns:
                    cols.append("nr_boost")

                tab = (
                    df_agg[cols]
                    .sort_values("expected", ascending=False)
                    .head(n).reset_index(drop=True)
                )

                lam = tab["expected"].to_numpy()
                tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]

                # Saat (başlangıç)
                start_iso_val = st.session_state.get("start_iso")
                try:
                    start_hh = pd.to_datetime(start_iso_val).strftime("%H:00") if start_iso_val else "-"
                except Exception:
                    start_hh = "-"
                tab["Saat"] = start_hh

                # CI95
                ci_vals = [poisson_ci(float(l)) for l in lam]
                tab["CI95"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in ci_vals]

                if "nr_boost" in tab.columns:
                    tab["NR"] = tab["nr_boost"].round(2)

                tab["E[olay] (λ)"] = tab["expected"].round(2)

                drop_cols = ["expected"]
                if "nr_boost" in tab.columns:
                    drop_cols.append("nr_boost")
                return tab.drop(columns=drop_cols)

            st.dataframe(top_risky_table(st.session_state["agg"]), use_container_width=True, height=300)

        st.subheader("Devriye özeti")
        if st.session_state.get("agg") is not None and btn_patrol:
            st.session_state["patrol"] = allocate_patrols(
                st.session_state["agg"], GEO_DF,
                k_planned=int(K_planned),
                duty_minutes=int(duty_minutes),
                cell_minutes=int(cell_minutes),
                travel_overhead=0.40
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        # Isı matrisi
        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            render_day_hour_heatmap(
                st.session_state["agg"],
                st.session_state["start_iso"],
                st.session_state["horizon_h"],
            )
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

        st.subheader("Dışa aktar")
        if st.session_state["agg"] is not None:
            csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "CSV indir", data=csv,
                file_name=f"risk_export_{int(time.time())}.csv",
                mime="text/csv"
            )

# ── Raporlar
else:
    st.header("Raporlar")
    st.info("Rapor sekmesi, mevcut koddan benzer şekilde taşınabilir.")
