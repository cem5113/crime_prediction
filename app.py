# app.py
from __future__ import annotations

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from utils.geo import load_geoid_layer, resolve_clicked_gid
from utils.forecast import precompute_base_intensity, aggregate_fast, prob_ge_k
from utils.patrol import allocate_patrols
from utils.ui import SMALL_UI_CSS, render_result_card, build_map_fast, render_kpi_row
from utils.hotspots import temp_hotspot_scores

# Isı matrisi: ayrı modül varsa oradan, yoksa ui'dan
try:
    from utils.heatmap import render_day_hour_heatmap
except ImportError:
    from utils.ui import render_day_hour_heatmap

# Pydeck yardımcıları: ayrı modülde olmalı; yoksa None
try:
    from utils.deck import build_map_fast_deck
except ImportError:
    build_map_fast_deck = None
  
from utils.constants import (
    SF_TZ_OFFSET, KEY_COL,
    MODEL_VERSION, MODEL_LAST_TRAIN,
    CATEGORIES,
)
from components.last_update import show_last_update_badge

# Basit CSV okuyucu (rapor modülüne ihtiyaç yok)
def load_events(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # Zaman sütununu esnekçe bul ve UTC'ye çevir
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in ["ts","timestamp","datetime","date_time","reported_at","occurred_at","time","date"]:
        if cand in lower:
            ts_col = lower[cand]
            break
    else:
        df["ts"] = pd.NaT
        return df.dropna(subset=["ts"])  # boş döner

    df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    # lat/lon isimlerini normalize et (opsiyonel)
    if "latitude" not in df.columns and "lat" in df.columns:
        df = df.rename(columns={"lat": "latitude"})
    if "longitude" not in df.columns and "lon" in df.columns:
        df = df.rename(columns={"lon": "longitude"})
    return df

    
# ── Sayfa ayarı: Streamlit'te en üstte olmalı
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)

# Yazıları (tablolar hariç) biraz büyüt
st.markdown("""
<style>
/* === Genel metin (tablo/datarame hariç) === */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
  font-size: 1.08rem;              /* arttırmak için 1.12rem, 1.15rem yapabilirsin */
}

/* Başlıklar */
h1 { font-size: 2.0rem; }
h2 { font-size: 1.60rem; }
h3 { font-size: 1.25rem; }

/* Sidebar metinleri */
section[data-testid="stSidebar"] * {
  font-size: 1.02rem;
}

/* Metric bileşeni */
[data-testid="stMetricLabel"] { font-size: 1.00rem; }
[data-testid="stMetricValue"] { font-size: 1.35rem; }

/* Buton ve radyo/checkbox etiketleri */
button, [data-baseweb="button"] { font-size: 1.00rem; }
div[role="radiogroup"] label p { font-size: 1.05rem; }

/* === Tabloları büyütme (varsayılan boyutta kalsın) === */
div[data-testid="stDataFrame"] *,
table[data-testid="stTable"] *,
.stTable * {
  font-size: 0.875rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Başlık ve "Son güncelleme" rozetini göster
st.title("SUTAM: Suç Tahmin Modeli")

try:
    events_df = load_events("data/events.csv")
    st.session_state["events_df"] = events_df if isinstance(events_df, pd.DataFrame) else None
    # "events" key'ini de aynı anda doldur (yoksa yarat)
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
sekme = st.sidebar.radio("", options=["Operasyon"], index=0, horizontal=True)
st.sidebar.divider()

# ---- GÜNCELLENEN KISIM ----
st.sidebar.header("Devriye Parametreleri")
engine = st.sidebar.radio("Harita motoru", ["Folium", "pydeck"], index=0, horizontal=True)
show_popups   = st.sidebar.checkbox("Hücre popup'larını (en olası 3 suç) göster", value=True)

# Harita katmanları
st.sidebar.subheader("Harita katmanları")
show_risk_layer   = st.sidebar.checkbox("Tahmin katmanı (risk)", value=True)
show_perm_hotspot = st.sidebar.checkbox("Sıcak nokta (kalıcı)", value=True)
show_temp_hotspot = st.sidebar.checkbox("Geçici sıcak nokta (son olaylar)", value=True)

hotspot_cat = st.sidebar.selectbox(
    "Hotspot kategorisi",
    options=["(Tüm suçlar)"] + CATEGORIES,
    index=0,
    help="Kalıcı/Geçici hotspot katmanları bu kategoriye göre gösterilir."
)

use_hot_hours = st.sidebar.checkbox("Geçici hotspot için gün içi saat filtresi", value=False)
hot_hours_rng = st.sidebar.slider("Saat aralığı (hotspot)", 0, 24, (0, 24), disabled=not use_hot_hours)

# Zaman ufku
now_local = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).replace(minute=0, second=0, microsecond=0)
ufuk = st.sidebar.radio(
    f"Zaman aralığı (başlangıç: {now_local:%Y-%m-%d %H:%M})",
    options=["24s", "48s", "7g"], index=0, horizontal=True
)
max_h, step = (24, 1) if ufuk == "24s" else (48, 3) if ufuk == "48s" else (7*24, 24)
start_h, end_h = st.sidebar.slider("Saat filtresi", min_value=0, max_value=max_h, value=(0, max_h), step=step)
start_dt_label = now_local + timedelta(hours=start_h)
end_dt_label   = now_local + timedelta(hours=end_h)
st.sidebar.caption(
    f"Seçilen aralık: {start_dt_label:%Y-%m-%d %H:%M} → {end_dt_label:%Y-%m-%d %H:%M}  ( {end_h - start_h} saat )"
)

# Kategori filtresi (tahmin motoru için)
sel_categories = st.sidebar.multiselect("Suç Tahmini için Kategori", ["(Hepsi)"] + CATEGORIES, default=[])
if sel_categories and "(Hepsi)" in sel_categories:
    filters = {"cats": CATEGORIES}
else:
    filters = {"cats": sel_categories or None}

show_advanced = st.sidebar.checkbox("Gelişmiş metrikleri göster (analist)", value=False)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned    = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)",   min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

# Aksiyon butonları (HATA BUNDAN GELİYORDU)
colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner")

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
            st.session_state["events_df"] = events_df 
            
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
                "agg": agg,
                "patrol": None,
                "start_iso": start_iso,
                "horizon_h": horizon_h,
                "events": events_df,  # 🔹 geçici hotspot için son olaylara ihtiyaç var
            })

            try:
                long_start_iso = (
                    datetime.utcnow()
                    + timedelta(hours=SF_TZ_OFFSET - 30*24)
                ).replace(minute=0, second=0, microsecond=0).isoformat()
            
                agg_long = aggregate_fast(
                    long_start_iso, 30*24, GEO_DF, BASE_INT,
                    events=events_df,          # local değişkeni kullan
                    near_repeat_alpha=0.0,     # referans için NR etkisini kapatmak isteyebilirsin
                    filters=None
                )
                st.session_state["agg_long"] = agg_long
            except Exception:
                st.session_state["agg_long"] = None

        agg = st.session_state["agg"]
        
        if isinstance(agg, pd.DataFrame):
            df_agg_for_map = agg if show_risk_layer else agg.iloc[0:0]
        else:
            df_agg_for_map = None
            
        events_all = st.session_state.get("events")
        lookback_h = int(np.clip(2 * st.session_state.get("horizon_h", 24), 24, 72))
        
        ev_recent_df = None
        if isinstance(events_all, pd.DataFrame) and not events_all.empty:
            ev_recent_df = events_all.copy()
            # zaman filtresi
            _ts = "ts" if "ts" in ev_recent_df.columns else ("timestamp" if "timestamp" in ev_recent_df.columns else None)
            ev_recent_df["ts"] = pd.to_datetime(ev_recent_df[_ts], utc=True, errors="coerce") if _ts else pd.NaT
            if "ts" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["ts"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h))]
            # kategori filtresi (eğer veri ‘type’ içeriyorsa)
            if hotspot_cat != "(Tüm suçlar)" and "type" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df[ev_recent_df["type"] == hotspot_cat]
            # gün içi saat filtresi
            if use_hot_hours and "ts" in ev_recent_df.columns:
                h1, h2 = hot_hours_rng[0], (hot_hours_rng[1] - 1) % 24
                ev_recent_df = ev_recent_df[ev_recent_df["ts"].dt.hour.between(h1, h2)]
            # lon/lat isimlerini normalize et
            if "latitude" not in ev_recent_df.columns and "lat" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lat": "latitude"})
            if "longitude" not in ev_recent_df.columns and "lon" in ev_recent_df.columns:
                ev_recent_df = ev_recent_df.rename(columns={"lon": "longitude"})
            ev_recent_df = ev_recent_df.dropna(subset=["latitude", "longitude"])
            if not ev_recent_df.empty:
                ev_recent_df["weight"] = 1.0

        # --- Grafik kapsamı için veri seti (df_plot) ---
        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            df_plot = ev_recent_df.copy()
        else:
            df_plot = pd.DataFrame(columns=["ts", "latitude", "longitude"])
                
        # --- Geçici hotspot HeatMap girdisi ---
        if isinstance(ev_recent_df, pd.DataFrame) and not ev_recent_df.empty:
            temp_points = ev_recent_df[["latitude", "longitude"]].copy()
            temp_points["weight"] = ev_recent_df["weight"] if "weight" in ev_recent_df.columns else 1.0
        else:
            temp_points = pd.DataFrame(columns=["latitude", "longitude", "weight"])
        
        # Katman kapalıysa boş gönder
        temp_points_effective = temp_points if show_temp_hotspot else pd.DataFrame(columns=["latitude","longitude","weight"])
                
        # ev_recent boşsa: üst risk hücrelerinden sentetik ısı üret (fallback)
        if show_temp_hotspot and temp_points_effective.empty and isinstance(df_agg_for_map, pd.DataFrame) and not df_agg_for_map.empty:
            topn = 80
            tmp = (
                df_agg_for_map.nlargest(topn, "expected")
                    .merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
                    .dropna(subset=["centroid_lat", "centroid_lon"])
            )
            temp_points_effective = tmp.rename(columns={"centroid_lat": "latitude", "centroid_lon": "longitude"})[
                ["latitude", "longitude"]
            ]
            temp_points_effective["weight"] = tmp["expected"].clip(lower=0).astype(float)
        
        # küçük sayaç (gösterge)
        st.sidebar.caption(f"Geçici hotspot noktası: {len(temp_points)}")

        df_agg_for_map = agg if show_risk_layer else agg.iloc[0:0]
        if agg is not None:
            if engine == "Folium":

                lookback_h = int(np.clip(2 * st.session_state.get("horizon_h", 24), 24, 72))
                
                source = st.session_state.get("events_df", None)
                if source is None:
                    try:
                        source = events_df  # aynı scope'ta varsa
                    except NameError:
                        source = None
                
                if isinstance(source, pd.DataFrame) and not source.empty:
                    ev_recent = source.copy()
                
                    # zaman kolonu (ts veya timestamp)
                    ts_col = "ts" if "ts" in ev_recent.columns else ("timestamp" if "timestamp" in ev_recent.columns else None)
                    if ts_col is None:
                        ev_recent = pd.DataFrame(columns=["latitude","longitude","weight"])  # kolon yoksa boş bırak
                    else:
                        ev_recent["timestamp"] = pd.to_datetime(ev_recent[ts_col], utc=True, errors="coerce")
                        ev_recent = ev_recent.dropna(subset=["timestamp"])
                
                        # koordinat kolonlarını normalize et (lat/lon -> latitude/longitude)
                        if "latitude" not in ev_recent.columns and "lat" in ev_recent.columns:
                            ev_recent = ev_recent.rename(columns={"lat": "latitude"})
                        if "longitude" not in ev_recent.columns and "lon" in ev_recent.columns:
                            ev_recent = ev_recent.rename(columns={"lon": "longitude"})
                
                        # son lookback_h saat filtresi
                        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=lookback_h)
                        ev_recent = ev_recent[
                            (ev_recent["timestamp"] >= cutoff)
                            & ev_recent["latitude"].notna()
                            & ev_recent["longitude"].notna()
                        ]
                
                        # ağırlık sütunu
                        if "weight" not in ev_recent.columns:
                            ev_recent["weight"] = 1.0
                else:
                    ev_recent = pd.DataFrame(columns=["latitude","longitude","weight"])
                
                m = build_map_fast(
                    df_agg=df_agg_for_map,
                    geo_features=GEO_FEATURES,
                    geo_df=GEO_DF,
                    show_popups=show_popups,
                    patrol=st.session_state.get("patrol"),
                
                    show_hotspot=show_perm_hotspot,
                    perm_hotspot_mode="heat",
                
                    show_temp_hotspot=show_temp_hotspot,
                    temp_hotspot_points=temp_points_effective,
                )

                import folium
                assert isinstance(m, folium.Map), f"st_folium beklediği tipte değil: {type(m)}"
        
                ret = st_folium(
                    m, key="riskmap",
                    width=1100,        # ← EN (px)
                    height=540,        # ← BOY (px) istersen sabit bırak
                    returned_objects=["last_object_clicked", "last_clicked"]
                )
                if ret:
                    gid, _ = resolve_clicked_gid(GEO_DF, ret)
                    if gid:
                        st.session_state["explain"] = {"geoid": gid}
        
            else:
                if build_map_fast_deck is None:
                    st.error("Pydeck harita modülü bulunamadı (utils/deck.py yüklenemedi). Lütfen Folium motorunu seçin.")
                    ret = None
                else:
                    deck = build_map_fast_deck(
                        df_agg_for_map if isinstance(df_agg_for_map, pd.DataFrame) else agg,  # elde ne varsa
                        GEO_DF,
                        show_poi=False,
                        show_transit=False,
                        patrol=st.session_state.get("patrol"),
                        show_hotspot=show_perm_hotspot,
                        show_temp_hotspot=show_temp_hotspot,
                        temp_hotspot_points=temp_points_effective,
                    )
                    st.pydeck_chart(deck)
                    # Not: pydeck tarafında tıklama yakalama ayrı yapılır.
                    ret = None
        
            # Açıklama kartı
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
        st.header("Risk Özeti")

        if st.session_state["agg"] is not None:
            a = st.session_state["agg"]
            kpi_expected = round(float(a["expected"].sum()), 2)
            high = int((a["tier"] == "Yüksek").sum())
            mid  = int((a["tier"] == "Orta").sum())
            low  = int((a["tier"] == "Hafif").sum())

            render_kpi_row([
                ("Beklenen Suç Sayısı", kpi_expected, "Seçili zaman ufkunda toplam beklenen olay sayısı"),
                ("Yüksek",               high,         "Yüksek öncelikli hücre sayısı"),
                ("Orta",                 mid,          "Orta öncelikli hücre sayısı"),
                ("Düşük",                low,          "Düşük öncelikli hücre sayısı"),
            ])
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")

        st.header("En riskli bölgeler")
        if st.session_state["agg"] is not None:

            def top_risky_table(df_agg: pd.DataFrame, n: int = 12, show_ci: bool = False) -> pd.DataFrame:
                # Poisson ~%95 güven aralığı (normal approx.)
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
            
                # 95% Güven Aralığı (isteğe bağlı)
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
            st.dataframe(
                top_risky_table(st.session_state["agg"], show_ci=show_advanced),
                use_container_width=True, height=300
            )
            if show_advanced:
                st.caption(
                    "95% Güven Aralığı: Aynı koşullar tekrarlansa, gerçek sayının ~%95 bu aralıkta kalması beklenir. "
                    "Hızlı hesap: λ ± 1.96·√λ (alt sınır 0'a kırpılır)."
                )

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
            render_day_hour_heatmap(st.session_state["agg"],
                                    st.session_state.get("start_iso"),
                                    st.session_state.get("horizon_h"))
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
if "events" not in st.session_state:
    st.session_state["events"] = pd.DataFrame({
        "ts": pd.date_range("2025-10-01", periods=50, freq="H"),
        "type": ["Theft"]*25 + ["Assault"]*25,
        "geoid": ["12345"]*50
    })



