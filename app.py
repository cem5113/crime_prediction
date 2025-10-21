# app.py  — SUTAM
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
    DISPLAY_CATEGORIES,          # UI'de görünen kategori başlıkları (Title Case)
    category_key_list,           # UI seçimini model anahtarlarına çevirir
)

# ── Artefakt içe aktarma & kanonik veri üretimi
from components.utils.loaders import import_latest_artifact, materialize_canonical

# ── Geo & hotspot yardımcıları
from components.utils.geo import load_geoid_layer, resolve_clicked_gid
from components.utils.hotspots import render_day_hour_heatmap

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
# ------------------------------------------------------------------
# Fallback: olay yükleyici (raporlar yoksa da çalışsın)
# ------------------------------------------------------------------
try:
    from reports import load_events  # components/utils/reports.py (varsa)
except Exception:
    def load_events(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
        lower = {str(c).strip().lower(): c for c in df.columns}
        for cand in ["ts", "timestamp", "datetime", "date_time", "reported_at", "occurred_at", "time", "date"]:
            if cand in lower:
                ts_col = lower[cand]
                break
        else:
            df["ts"] = pd.NaT
            return df.dropna(subset=["ts"])
        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        if "latitude" not in df.columns and "lat" in df.columns:
            df = df.rename(columns={"lat": "latitude"})
        if "longitude" not in df.columns and "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude"})
        return df

# ------------------------------------------------------------------
# Streamlit temel ayarlar
# ------------------------------------------------------------------
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)
st.title("SUTAM: Suç Tahmin Modeli")

# Olay verisi (opsiyonel)
events_csv_path = os.path.join(DATA_DIR, "events.csv")  # istersen data/events.csv yap
try:
    events_df = load_events(events_csv_path)
    st.session_state["events_df"] = events_df if isinstance(events_df, pd.DataFrame) else None
    st.session_state["events"] = st.session_state["events_df"]
    if isinstance(events_df, pd.DataFrame) and not events_df.empty and "ts" in events_df.columns:
        data_upto_val = pd.to_datetime(events_df["ts"]).max().date().isoformat()
    else:
        data_upto_val = None
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
geojson_path = os.path.join(DATA_DIR, "sf_cells.geojson")  # ✅ doğru konum
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
        paths = materialize_canonical(out["sf"], out["fr"])
        # önemli: grid dosyası artefaktan güncellenmiş olabilir → tüm cache’i temizle
        st.cache_data.clear()
        st.success("Artefakt içe aktarıldı, canonical veri üretildi ve önbellek temizlendi.")
        st.caption(f"Üretilen dosyalar: {paths}")
        # küçük bir ipucu: kullanıcıya üstte 'Tahmin et' çalıştırmasını hatırlat
        st.toast("Hazır! Yeni veriyle haritayı güncellemek için ‘Tahmin et’ tuşuna basın.", icon="✅")
    except Exception as e:
        st.error(f"İçe aktarma/kanonikleştirme hatası: {e}")

# Harita motoru
st.sidebar.header("Görselleştirme")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)

# Harita katmanları
st.sidebar.subheader("Harita katmanları")
show_poi      = st.sidebar.checkbox("POI overlay", value=False)
show_transit  = st.sidebar.checkbox("Toplu taşıma overlay", value=False)
show_popups   = st.sidebar.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)

# Grafik kapsamı
scope = st.sidebar.radio("Grafik kapsamı", ["Tüm şehir", "Seçili hücre"], index=0)

# Hotspot ayarları
show_hotspot        = True
show_temp_hotspot   = True
hotspot_cat = st.sidebar.selectbox(
    "Hotspot kategorisi",
    options=["(Tüm suçlar)"] + DISPLAY_CATEGORIES,  # ← Title Case liste
    index=0
)
use_hot_hours = st.sidebar.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
hot_hours_rng = st.sidebar.slider("Saat aralığı (hotspot)", 0, 24, (0, 24), disabled=not use_hot_hours)

# Zaman ufku
ufuk = st.sidebar.radio("Zaman Aralığı (şimdiden)", options=["24s", "48s", "7g"], index=0, horizontal=True)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7*24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", min_value=0, max_value=max_h, value=(0, max_h), step=step)

# 🎯 Kategori filtresi (UI → model anahtarları)
sel_display_cats = st.sidebar.multiselect(
    "Kategori",
    ["(Hepsi)"] + DISPLAY_CATEGORIES,
    default=[]
)

if sel_display_cats and "(Hepsi)" in sel_display_cats:
    # Hepsi seçildiyse tüm anahtarları topla
    selected_keys = []
    for disp in DISPLAY_CATEGORIES:
        selected_keys.extend(category_key_list(disp))
else:
    selected_keys = []
    for disp in sel_display_cats:
        selected_keys.extend(category_key_list(disp))

# Tahmin/aggregate fonksiyonlarına geçecek filtre nesnesi
filters = {"cats": (selected_keys or None)}

# Analist görünümü
show_advanced = st.sidebar.checkbox("Gelişmiş metrikleri göster (analist)", value=False)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned    = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)",   min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et", use_container_width=True)
btn_patrol  = colB.button("Devriye öner", use_container_width=True)

# ------------------------------------------------------------------
# State init
# ------------------------------------------------------------------
if "agg" not in st.session_state:
    st.session_state.update({"agg": None, "patrol": None, "start_iso": None, "horizon_h": None, "explain": None})

# ------------------------------------------------------------------
# Operasyon
# ------------------------------------------------------------------
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")

        if btn_predict or st.session_state["agg"] is None:
            start_dt  = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
            horizon_h = max(1, end_h - start_h)
            start_iso = start_dt.isoformat()

            events_df = load_events(events_csv_path)
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
            st.session_state.update({"agg": agg, "patrol": None, "start_iso": start_iso, "horizon_h": horizon_h, "events": events_df})

            try:
                long_start_iso = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET - 30*24)).replace(minute=0, second=0, microsecond=0).isoformat()
                agg_long = aggregate_fast(long_start_iso, 30*24, GEO_DF, BASE_INT, events=events_df, near_repeat_alpha=0.0, filters=None)
                st.session_state["agg_long"] = agg_long
            except Exception:
                st.session_state["agg_long"] = None

        agg = st.session_state["agg"]

        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * st.session_state.get("horizon_h", 24), 24, 72))

        ev_recent_df = None
        if isinstance(events_all, pd.DataFrame) and not events_all.empty:
            ev_recent_df = events_all.copy()
            _ts = "ts" if "ts" in ev_recent_df.columns else ("timestamp" if "timestamp" in ev_recent_df.columns else None)
            ev_recent_df["ts"] = pd.to_datetime(ev_recent_df[_ts], utc=True, errors="coerce") if _ts else pd.NaT
            if "ts" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h))]
            if hotspot_cat != "(Tüm suçlar)" and "type" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["type"] == hotspot_cat]
            if use_hot_hours and "ts" in ev_recent_df.columns:
                h1, h2 = hot_hours_rng[0], (hot_hours_rng[1] - 1) % 24
                ev_recent_df = ev_recent_df[ev_recent_df["ts"].dt.hour.between(h1, h2)]
            if "latitude" not in ev_recent_df.columns and "lat" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lat": "latitude"})
            if "longitude" not in ev_recent_df.columns and "lon" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lon": "longitude"})
            ev_recent_df = ev_recent_df.dropna(subset=["latitude", "longitude"])
            if not ev_recent_df.empty:
                ev_recent_df["weight"] = 1.0

        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            keep_cols = [c for c in ["ts", "latitude", "longitude", KEY_COL] if c in ev_recent_df.columns]
            df_plot = ev_recent_df[keep_cols].copy()
        else:
            df_plot = pd.DataFrame(columns=["ts", "latitude", "longitude"])

        if scope == "Seçili hücre" and st.session_state.get("explain", {}).get("geoid"):
            gid = str(st.session_state["explain"]["geoid"])
            if KEY_COL in df_plot.columns:
                df_plot = df_plot[df_plot[KEY_COL].astype(str) == gid]

        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            temp_points = ev_recent_df[["latitude", "longitude"]].copy()
            temp_points["weight"] = ev_recent_df["weight"] if "weight" in ev_recent_df.columns else 1.0
        else:
            temp_points = pd.DataFrame(columns=["latitude", "longitude", "weight"])

        if show_temp_hotspot and temp_points.empty and isinstance(agg, pd.DataFrame) and not agg.empty:
            topn = 80
            tmp = (
                agg.nlargest(topn, "expected")
                   .merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
                   .dropna(subset=["centroid_lat", "centroid_lon"])
            )
            temp_points = tmp.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})[["latitude", "longitude"]]
            temp_points["weight"] = tmp["expected"].clip(lower=0).astype(float)

        st.sidebar.caption(f"Geçici hotspot noktası: {len(temp_points)}")

        if agg is not None:
            if engine == "Folium":
                source = st.session_state.get("events_df", None)
                if isinstance(source, pd.DataFrame) and not source.empty:
                    ev_recent = source.copy()
                    ts_col = "ts" if "ts" in ev_recent.columns else ("timestamp" if "timestamp" in ev_recent.columns else None)
                    if ts_col is None:
                        ev_recent = pd.DataFrame(columns=["latitude","longitude","weight"])
                    else:
                        ev_recent["timestamp"] = pd.to_datetime(ev_recent[ts_col], utc=True, errors="coerce")
                        ev_recent = ev_recent.dropna(subset=["timestamp"])
                        if "latitude" not in ev_recent.columns and "lat" in ev_recent.columns:
                            ev_recent = ev_recent.rename(columns={"lat": "latitude"})
                        if "longitude" not in ev_recent.columns and "lon" in ev_recent.columns:
                            ev_recent = ev_recent.rename(columns={"lon": "longitude"})
                        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h)
                        ev_recent = ev_recent[(ev_recent["timestamp"] >= cutoff) & ev_recent["latitude"].notna() & ev_recent["longitude"].notna()]
                        if "weight" not in ev_recent.columns:
                            ev_recent["weight"] = 1.0
                else:
                    ev_recent = pd.DataFrame(columns=["latitude","longitude","weight"])

                m = build_map_fast(
                    df_agg=agg,
                    geo_features=GEO_FEATURES,
                    geo_df=GEO_DF,
                    show_popups=show_popups,
                    patrol=st.session_state.get("patrol"),
                    show_poi=show_poi,
                    show_transit=show_transit,
                    show_hotspot=show_hotspot,
                    perm_hotspot_mode="heat",
                    show_temp_hotspot=show_temp_hotspot,
                    temp_hotspot_points=temp_points,
                )

                import folium
                assert isinstance(m, folium.Map), f"st_folium beklediği tipte değil: {type(m)}"
                ret = st_folium(m, key="riskmap", height=540, returned_objects=["last_object_clicked", "last_clicked"])
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (components/utils/deck.py). Lütfen Folium motorunu seçin.")
                    ret = None
                else:
                    deck = build_map_fast_deck(
                        agg, GEO_DF,
                        show_poi=show_poi, show_transit=show_transit,
                        patrol=st.session_state.get("patrol"),
                        show_hotspot=show_hotspot,
                        show_temp_hotspot=show_temp_hotspot,
                        temp_hotspot_points=temp_points,
                    )
                    st.pydeck_chart(deck)
                    ret = None

            start_iso  = st.session_state["start_iso"]
            horizon_h  = st.session_state["horizon_h"]
            info = st.session_state.get("explain")
            if info and info.get("geoid"):
                render_result_card(agg, info["geoid"], start_iso, horizon_h)
            else:
                st.info("Haritada bir hücreye tıklayın; kart burada görünecek.")
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
            def top_risky_table(df_agg: pd.DataFrame, n: int = 12, show_ci: bool = False) -> pd.DataFrame:
                def poisson_ci(lam: float, z: float = 1.96) -> tuple[float, float]:
                    s = float(np.sqrt(max(lam, 1e-9)))
                    return max(0.0, lam - z * s), lam + z * s
                cols = [KEY_COL, "expected"]
                if "nr_boost" in df_agg.columns:
                    cols.append("nr_boost")
                tab = df_agg[cols].sort_values("expected", ascending=False).head(n).reset_index(drop=True)
                lam = tab["expected"].to_numpy()
                from math import exp
                def prob_ge_k(lmbd, k):  # güvenli mini fallback
                    from math import exp
                    return 1 - exp(-lmbd) if k == 1 else 1.0
                try:
                    tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]
                except Exception:
                    tab["P(≥1)%"] = [round((1 - np.exp(-l)) * 100, 1) for l in lam]
                start_iso_val = st.session_state.get("start_iso")
                try:
                    start_hh = pd.to_datetime(start_iso_val).strftime("%H:00") if start_iso_val else "-"
                except Exception:
                    start_hh = "-"
                tab["Saat"] = start_hh
                if show_ci:
                    ci_vals = [poisson_ci(float(l)) for l in lam]
                    tab["95% Güven Aralığı"] = [f"[{lo:.2f}, {hi:.2f}]" for lo, hi in ci_vals]
                if "nr_boost" in tab.columns:
                    tab["NR"] = tab["nr_boost"].round(2)
                tab["E[olay] (λ)"] = tab["expected"].round(2)
                drop_cols = ["expected"]
                if "nr_boost" in tab.columns:
                    drop_cols.append("nr_boost")
                return tab.drop(columns=drop_cols)
            st.dataframe(top_risky_table(st.session_state["agg"], show_ci=show_advanced), use_container_width=True, height=300)
            if show_advanced:
                st.caption("95% CI ≈ λ ± 1.96·√λ (alt sınır 0’a kırpılır).")

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

        st.subheader("Gün × Saat Isı Matrisi")
        if st.session_state.get("agg") is not None and st.session_state.get("start_iso"):
            render_day_hour_heatmap(st.session_state["agg"], st.session_state.get("start_iso"), st.session_state.get("horizon_h"))
        else:
            st.caption("Isı matrisi, bir tahmin üretildiğinde gösterilir.")

        st.subheader("Dışa aktar")
        if st.session_state["agg"] is not None:
            csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
            st.download_button("CSV indir", data=csv, file_name=f"risk_export_{int(time.time())}.csv", mime="text/csv")

# ------------------------------------------------------------------
# Raporlar
# ------------------------------------------------------------------
elif sekme == "Raporlar":
    agg_current = st.session_state.get("agg")
    agg_long    = st.session_state.get("agg_long")
    events_src  = st.session_state.get("events")
    if not isinstance(events_src, pd.DataFrame) or events_src.empty:
        events_src = st.session_state.get("events_df")
    render_reports(events_df=events_src, agg_current=agg_current, agg_long_term=agg_long)
