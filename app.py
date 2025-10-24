# app.py — SUTAM (revize tam sürüm)
from __future__ import annotations

import os, time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ── Tek doğruluk kaynağı: constants
from components.utils.constants import (
    SF_TZ_OFFSET, KEY_COL,
    MODEL_VERSION, MODEL_LAST_TRAIN,
    DISPLAY_CATEGORIES,
    category_key_list,
    DATA_DIR,                    # ✅ EKLENDİ
)

# ── Artefakt içe aktarma & kanonik veri üretimi
from components.utils.loaders import import_latest_artifact, materialize_canonical

# ── Geo & hotspot yardımcıları
from components.utils.geo import load_geoid_layer, resolve_clicked_gid
from components.utils.hotspots import render_day_hour_heatmap
from components.utils.constants import GRID_FILE

# ── Tahmin & devriye yardımcıları
from components.utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from components.utils.patrol import allocate_patrols

# ── UI yardımcıları
from components.utils.layout_utils import SMALL_UI_CSS, render_result_card, build_map_fast, render_kpi_row

# ── Pydeck (opsiyonel)
try:
    from components.utils.deck import build_map_fast_deck
except Exception:
    build_map_fast_deck = None

# ── Son güncelleme rozeti
from components.last_update import show_last_update_badge

# ── Raporlar sekmesi (varsa)
try:
    from components.ui.reports import render_reports
    HAS_REPORTS = True
except Exception:
    HAS_REPORTS = False
    def render_reports(**kwargs):
        st.info("Raporlar modülü bulunamadı (components/ui/reports.py).")

geojson_candidates = [
    # secrets ile gelmiş olabilir
    os.environ.get("GRID_FILE"),
    # constants içindeki canonical (Path objesi olabilir)
    str(GRID_FILE) if "GRID_FILE" in dir() and GRID_FILE else None,
    # repo içi klasik konumlar
    os.path.join("components", "data", "sf_cells.geojson"),
    os.path.join("data", "sf_cells.geojson"),
]

GEO_DF, GEO_FEATURES, GEO_DEBUG = load_geoid_layer_any(geojson_candidates, return_debug=True)

if GEO_DF.empty:
    import streamlit as st
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.caption("Denediğim yollar ve teşhis:")
    st.code("\n".join([d for d in GEO_DEBUG if d]), language="text")
    st.stop()

# ------------------------------------------------------------------
# Fallback: olay yükleyici — Parquet öncelikli
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Fallback: olay yükleyici (raporlar yoksa da çalışsın) — Parquet öncelikli
# ------------------------------------------------------------------
def load_events(path: str) -> pd.DataFrame:
    import os
    import pandas as pd
    p = os.fspath(path)
    try:
        if p.lower().endswith(".parquet") or os.path.splitext(p)[1].lower() == ".parquet":
            df = pd.read_parquet(p)
        else:
            # hem parquet hem csv dene (önce parquet)
            pq = os.path.splitext(p)[0] + ".parquet"
            if os.path.exists(pq):
                df = pd.read_parquet(pq)
            else:
                df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

    lower = {str(c).strip().lower(): c for c in df.columns}
    # ts kolonunu normalize et
    ts_col = None
    for cand in ["ts", "timestamp", "datetime", "date_time", "reported_at", "occurred_at", "time", "date"]:
        if cand in lower:
            ts_col = lower[cand]; break
    if ts_col:
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
    else:
        df["ts"] = pd.NaT
        df = df.dropna(subset=["ts"])

    # koordinat adlarını normalize et
    if "latitude" not in df.columns and "lat" in lower:
        df = df.rename(columns={lower["lat"]: "latitude"})
    if "longitude" not in df.columns and "lon" in lower:
        df = df.rename(columns={lower["lon"]: "longitude"})
    return df

# Olay verisi (opsiyonel)
events_parquet_path = os.path.join(DATA_DIR, "events.parquet")  # ✅ parquet bekler, csv fallback
try:
    events_df = load_events(events_parquet_path)
    st.session_state["events_df"] = events_df if isinstance(events_df, pd.DataFrame) else None
    st.session_state["events"] = st.session_state["events_df"]
    data_upto_val = (
        pd.to_datetime(events_df["ts"]).max().date().isoformat()
        if isinstance(events_df, pd.DataFrame) and not events_df.empty and "ts" in events_df.columns
        else None
    )
except Exception:
    st.session_state["events_df"] = None
    st.session_state["events"] = None
    data_upto_val = None

# ------------------------------------------------------------------
# Streamlit temel ayarlar
# ------------------------------------------------------------------
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("SUTAM: Suç Tahmin Modeli")

# Olay verisi (opsiyonel)
events_parquet_path = os.path.join(DATA_DIR, "events.parquet")
try:
    events_df = load_events(events_parquet_path)
    st.session_state["events_df"] = events_df if isinstance(events_df, pd.DataFrame) else None
    st.session_state["events"] = st.session_state["events_df"]
    data_upto_val = (
        pd.to_datetime(events_df["ts"]).max().date().isoformat()
        if isinstance(events_df, pd.DataFrame) and not events_df.empty and "ts" in events_df.columns
        else None
    )
except Exception:
    st.session_state["events_df"] = None
    st.session_state["events"] = None
    data_upto_val = None

show_last_update_badge(
    data_upto=data_upto_val,
    model_version=MODEL_VERSION,
    last_train=MODEL_LAST_TRAIN,
)

# ------------------------------------------------------------------
# GEO katmanı
# ------------------------------------------------------------------
geojson_path = os.path.join(DATA_DIR, "sf_cells.geojson")
GEO_DF, GEO_FEATURES = load_geoid_layer(geojson_path)
if GEO_DF.empty:
    st.error("GEOJSON yüklenemedi veya satır yok.")
    st.stop()

# Model tabanı
BASE_INT = precompute_base_intensity(GEO_DF)


def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")


# ---------------- Sidebar ----------------
st.sidebar.markdown("### Görünüm")
sekme_options = ["Operasyon"]
if HAS_REPORTS:
    sekme_options.append("Raporlar")
sekme = st.sidebar.radio("", options=sekme_options, index=0, horizontal=True)
st.sidebar.divider()

# 📦 Artefakt içe aktar & yenile
if st.sidebar.button("📦 Artefaktı içe aktar & yenile", use_container_width=True):
    try:
        out = import_latest_artifact(save_raw=False)
        paths = materialize_canonical(out.sf, out.fr, pred_opt=out.pred_opt)
        st.cache_data.clear()
        msg = "Artefakt içe aktarıldı, canonical veri üretildi ve önbellek temizlendi."
        if getattr(out, "grid_updated", False):
            msg += " (Grid güncellendi)"
        st.success(msg)
        st.caption(f"Üretilen dosyalar: {paths}")
        st.toast("Hazır! Yeni veriyle haritayı güncellemek için ‘Tahmin et’ tuşuna basın.", icon="✅")
    except Exception as e:
        st.error(f"İçe aktarma/kanonikleştirme hatası: {e}")


# Harita motoru ve filtreler
st.sidebar.header("Görselleştirme")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)
show_poi = st.sidebar.checkbox("POI overlay", value=False)
show_transit = st.sidebar.checkbox("Toplu taşıma overlay", value=False)
show_popups = st.sidebar.checkbox("Hücre popup'larını göster", value=True)
scope = st.sidebar.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0)
show_hotspot = True
show_temp_hotspot = True
hotspot_cat = st.sidebar.selectbox("Hotspot kategorisi", ["(Tüm suçlar)"] + DISPLAY_CATEGORIES, index=0)
use_hot_hours = st.sidebar.checkbox("Geçici hotspot saat filtresi", value=False)
hot_hours_rng = st.sidebar.slider("Saat aralığı (hotspot)", 0, 24, (0, 24), disabled=not use_hot_hours)
ufuk = st.sidebar.radio("Zaman Aralığı", ["24s", "48s", "7g"], index=0, horizontal=True)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7 * 24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", 0, max_h, (0, max_h), step=step)
sel_display_cats = st.sidebar.multiselect("Kategori", ["(Hepsi)"] + DISPLAY_CATEGORIES, default=[])

if sel_display_cats and "(Hepsi)" in sel_display_cats:
    selected_keys = [k for disp in DISPLAY_CATEGORIES for k in category_key_list(disp)]
else:
    selected_keys = [k for disp in sel_display_cats for k in category_key_list(disp)]
filters = {"cats": (selected_keys or None)}

show_advanced = st.sidebar.checkbox("Gelişmiş metrikleri göster", value=False)
st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned = st.sidebar.number_input("Planlanan devriye (K)", 1, 50, 6)
duty_minutes = st.sidebar.number_input("Görev süresi (dk)", 15, 600, 120)
cell_minutes = st.sidebar.number_input("Hücre başına kontrol (dk)", 2, 30, 6)
colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et", use_container_width=True)
btn_patrol = colB.button("Devriye öner", use_container_width=True)

# ------------------------------------------------------------------
# State init
# ------------------------------------------------------------------
if "agg" not in st.session_state:
    st.session_state.update({"agg": None, "patrol": None, "start_iso": None, "horizon_h": None, "explain": None})


# ------------------------------------------------------------------
# Operasyon sekmesi
# ------------------------------------------------------------------
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])
    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")

        if btn_predict or st.session_state["agg"] is None:
            start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0)
            horizon_h = max(1, end_h - start_h)
            start_iso = start_dt.isoformat()
            events_df = load_events(events_parquet_path)
            st.session_state["events_df"] = events_df
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
                "agg": agg, "patrol": None,
                "start_iso": start_iso, "horizon_h": horizon_h, "events": events_df
            })
            try:
                long_start_iso = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30 * 24)).replace(minute=0, second=0).isoformat()
                agg_long = aggregate_fast(long_start_iso, 30 * 24, GEO_DF, BASE_INT, events=events_df, near_repeat_alpha=0.0)
                st.session_state["agg_long"] = agg_long
            except Exception:
                st.session_state["agg_long"] = None

        agg = st.session_state["agg"]

        if agg is not None:
            if engine == "Folium":
                m = build_map_fast(
                    df_agg=agg, geo_features=GEO_FEATURES, geo_df=GEO_DF,
                    show_popups=show_popups,
                    patrol=st.session_state.get("patrol"),
                    show_poi=show_poi, show_transit=show_transit,
                    show_hotspot=show_hotspot, perm_hotspot_mode="heat",
                    show_temp_hotspot=show_temp_hotspot,
                    temp_hotspot_points=pd.DataFrame()
                )
                import folium
                assert isinstance(m, folium.Map)
                ret = st_folium(m, key="riskmap", height=540, returned_objects=["last_clicked"])
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck modülü yok, Folium'u seçin.")
                else:
                    deck = build_map_fast_deck(agg, GEO_DF)
                    st.pydeck_chart(deck)

            info = st.session_state.get("explain")
            if info and info.get("geoid"):
                render_result_card(agg, info["geoid"], st.session_state["start_iso"], st.session_state["horizon_h"])
            else:
                st.info("Haritada bir hücreye tıklayın.")
        else:
            st.info("Önce ‘Tahmin et’ ile tahmin üretin.")

    with col2:
        st.subheader("Risk Özeti")
        if st.session_state["agg"] is not None:
            a = st.session_state["agg"]
            kpi_expected = round(float(a["expected"].sum()), 2)
            high = int((a["tier"] == "Yüksek").sum())
            mid = int((a["tier"] == "Orta").sum())
            low = int((a["tier"] == "Hafif").sum())
            render_kpi_row([
                ("Beklenen olay", kpi_expected, "Seçili ufuk toplam olay"),
                ("Yüksek", high, "Yüksek öncelikli hücre"),
                ("Orta", mid, "Orta öncelikli hücre"),
                ("Düşük", low, "Düşük öncelikli hücre"),
            ])
        else:
            st.info("Tahmin yok.")

        st.subheader("En riskli bölgeler")
        if st.session_state["agg"] is not None:
            tab = st.session_state["agg"][[KEY_COL, "expected"]].sort_values("expected", ascending=False).head(12)
            tab["P(≥1)%"] = (1 - np.exp(-tab["expected"])) * 100
            st.dataframe(tab, use_container_width=True, height=300)

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
                "zone": z["id"], "cells": z["planned_cells"],
                "eta_min": z["eta_minutes"], "util_%": z["utilization_pct"],
                "avg_risk": round(z["expected_risk"], 2),
            } for z in patrol["zones"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

        st.subheader("Dışa aktar")
        if st.session_state["agg"] is not None:
            csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
            st.download_button("CSV indir", data=csv, file_name=f"risk_{int(time.time())}.csv", mime="text/csv")
            pq = st.session_state["agg"].to_parquet(index=False)
            st.download_button("Parquet indir", data=pq, file_name=f"risk_{int(time.time())}.parquet", mime="application/octet-stream")

# ------------------------------------------------------------------
# Raporlar sekmesi
# ------------------------------------------------------------------
elif sekme == "Raporlar":
    agg_current = st.session_state.get("agg")
    agg_long = st.session_state.get("agg_long")
    events_src = st.session_state.get("events") or st.session_state.get("events_df")
    render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)
