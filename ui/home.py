# home.py
import os
import json
import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ----------------------------
# Config / Constants
# ----------------------------
APP_NAME = os.getenv("APP_NAME", "SUTAM – Suç Tahmin Modeli")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.3.1")
DAILY_UPDATE_HOUR_SF = int(os.getenv("DAILY_UPDATE_HOUR_SF", "19"))  # SF saati
CITY_CENTER = [-122.4194, 37.7749]  # SF merkez
DEFAULT_ZOOM = 11.0

# ----------------------------
# Time helpers (fallback)
# ----------------------------
def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def now_tz(tzname: str):
    try:
        import pytz
        return dt.datetime.now(pytz.timezone(tzname))
    except Exception:
        # Fallback: UTC
        return now_utc()

def now_sf():
    return now_tz("America/Los_Angeles")

def now_tr():
    return now_tz("Europe/Istanbul")

def fmt(dtobj):
    # 2025-10-20 16:45 gibi
    return dtobj.strftime("%Y-%m-%d %H:%M")

# ----------------------------
# Data loaders (swap with your dataio/loaders.py if available)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=900)
def load_dim_cells(path="data/dim_cells.geojson"):
    # GeoJSON veya GeoParquet olabilir; örnek GeoJSON
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    # properties.geoid ve properties.neighborhood beklenir
    return gj

@st.cache_data(show_spinner=False, ttl=900)
def load_predictions(time_scope="24h", path="data/pred_cell_timeslices.parquet"):
    # Beklenen kolonlar: geoid, risk_score (0–1), pred_expected (float)
    df = pd.read_parquet(path)
    # Son 24 saatlik dilim veya en güncel saat seçimi
    if time_scope == "24h" and "timeslice_start" in df.columns:
        cutoff = now_utc() - dt.timedelta(hours=24)
        df = df[df["timeslice_start"] >= pd.Timestamp(cutoff)]
    # Hücre başına son değer (en güncel) — UI için sadeleştirme
    if "timeslice_start" in df.columns:
        df = (df.sort_values("timeslice_start")
                .groupby("geoid", as_index=False)
                .tail(1))
    # Eksik kolonlar için emniyet
    df["risk_score"] = df.get("risk_score", pd.Series(np.clip(df.get("pred_expected", 0)/2.0, 0, 1)))
    df["pred_expected"] = df.get("pred_expected", 0.0)
    return df[["geoid", "risk_score", "pred_expected"]].copy()

# ----------------------------
# Metrics (swap with services/metrics.py)
# ----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_kpis():
    # Günlük ölçümler – gerçek metrik servisine bağlayın
    return {
        "AUC (günlük)": 0.83,
        "HitRate@TopK": "74%",
        "Brier": 0.17,
    }

# ----------------------------
# Map helpers
# ----------------------------
def join_geojson_with_scores(geojson, scores_df):
    # GeoJSON Feature.properties.geoid ile join
    score_map = scores_df.set_index("geoid").to_dict("index")
    for feat in geojson["features"]:
        g = str(feat["properties"].get("geoid"))
        vals = score_map.get(g) or score_map.get(int(g)) if g and g.isdigit() else score_map.get(g)
        if vals:
            feat["properties"]["risk_score"] = float(vals["risk_score"])
            feat["properties"]["pred_expected"] = float(vals["pred_expected"])
        else:
            feat["properties"]["risk_score"] = 0.0
            feat["properties"]["pred_expected"] = 0.0
    return geojson

def risk_color_expression():
    # 0-1 risk_score -> [r,g,b,alpha] (PyDeck JS expression)
    # Basit palet: düşük yeşil, orta sarı, yüksek kırmızı
    return """
    [
      risk_score * 255,
      (1 - Math.abs(risk_score - 0.5)*2) * 255,
      (1 - risk_score) * 255,
      140
    ]
    """

def make_deck(geojson):
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color=risk_color_expression(),
        get_line_color=[60,60,60,100],
        line_width_min_pixels=1,
    )
    tooltip = {
        "html": "<b>GEOID:</b> {geoid}<br/>"
                "<b>Mahalle:</b> {neighborhood}<br/>"
                "<b>E[olay]:</b> {pred_expected}<br/>"
                "<b>Risk:</b> {risk_score}",
        "style": {"background": "rgba(22,22,22,0.9)", "color": "white"}
    }
    view = pdk.ViewState(latitude=CITY_CENTER[1], longitude=CITY_CENTER[0], zoom=DEFAULT_ZOOM)
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip), layer

# ----------------------------
# UI
# ----------------------------
def render_badge_row():
    sf_now = now_sf()
    tr_now = now_tr()
    last_trained = os.getenv("LAST_TRAINED_AT_SF", "—")
    data_upto = f"{DAILY_UPDATE_HOUR_SF}:00 (SF)"
    cols = st.columns([2,1])
    with cols[0]:
        st.markdown(f"### {APP_NAME}")
        st.caption(
            f"**Model:** {MODEL_VERSION} • **Son eğitim (SF):** {last_trained} • "
            f"**Güncelleme saati:** {data_upto} • "
            f"**Şimdi (SF):** {fmt(sf_now)} • **(TR):** {fmt(tr_now)}"
        )
    with cols[1]:
        # Sadece YENİLE butonu (kullanıcının isteği)
        if st.button("🔄 Veriyi Yenile", use_container_width=True):
            st.cache_data.clear()
            st.session_state["last_reload_at_sf"] = fmt(now_sf())
            st.success("Veri önbelleği temizlendi ve yeniden yüklendi.")

def render_kpis(kpis: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("AUC (günlük)", kpis.get("AUC (günlük)", "—"))
    with c2:
        st.metric("HitRate@TopK", kpis.get("HitRate@TopK", "—"))
    with c3:
        st.metric("Brier", kpis.get("Brier", "—"))

def render_mini_map():
    st.subheader("Mini Risk Haritası (0–24 saat)")
    try:
        gj = load_dim_cells()  # data/dim_cells.geojson
        df = load_predictions(time_scope="24h")  # data/pred_cell_timeslices.parquet
        gj = join_geojson_with_scores(gj, df)
        deck, _ = make_deck(gj)
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as e:
        st.warning(f"Harita yüklenemedi: {e}")

def main():
    st.set_page_config(page_title="Ana Sayfa", layout="wide")
    render_badge_row()
    st.divider()
    render_kpis(load_kpis())
    st.divider()
    render_mini_map()

if __name__ == "__main__":
    main()
