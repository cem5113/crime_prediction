# utils/ui.py
from __future__ import annotations
import math, json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import folium
import streamlit as st

from utils.constants import KEY_COL, CRIME_TYPES
from utils.forecast import pois_pi90

# ────────────────────────────── KÜÇÜK VE TUTARLI TİPOGRAFİ ──────────────────────────────
SMALL_UI_CSS = """
<style>
/* === GENEL: tüm yazılar küçük, satır aralığı dar === */
html, body, [class*="css"] { font-size: 12px; line-height: 1.28; }

/* === Başlıklar (yalnızca H1 büyük) === */
h1 { font-size: 1.9rem; line-height: 1.2; margin: .45rem 0 .35rem 0; }
h2 { font-size: .95rem;  margin: .25rem 0; }
h3 { font-size: .88rem;  margin: .18rem 0; }

/* === İç boşlukları sıkılaştır === */
section.main > div.block-container { padding-top: .55rem; padding-bottom: .10rem; }
[data-testid="stSidebar"] .block-container { padding-top: .25rem; padding-bottom: .25rem; }
div.element-container { margin-bottom: .22rem; }

/* === Form/label/yardım metinleri === */
label, .stMarkdown p, .stCaption, .stText, .stRadio, .stSelectbox, .stNumberInput { font-size: .82rem; }
small, .stCaption, .st-emotion-cache-1wbqy5l { font-size: .74rem; }

/* === Butonlar === */
.stButton > button,
.stDownloadButton > button {
  font-size: .80rem;
  padding: 4px 10px;
  border-radius: 8px;
}

/* === Slider & input içerikleri === */
[data-testid="stSlider"] { padding-top: .10rem; padding-bottom: .05rem; }
input, textarea { font-size: .80rem !important; }

/* === Metric kartları (genel) === */
[data-testid="stMetricValue"] { font-size: .95rem; }
[data-testid="stMetricLabel"] { font-size: .68rem; color:#666; }
[data-testid="stMetric"]      { padding: .06rem 0 .02rem 0; }

/* st.metric ellipsis düzeltmesi (label kesilmesin) */
[data-testid="stMetricLabel"] p{
  max-width:none !important;
  overflow:visible !important;
  text-overflow:clip !important;
  white-space:nowrap !important;
  margin:0 !important;
}

/* Risk Özeti bloğu (bir tık daha küçük) */
#risk-ozet [data-testid="stMetricValue"] { font-size: .90rem; line-height: 1.0; }
#risk-ozet [data-testid="stMetricLabel"] { font-size: .64rem; color:#6b7280; }
#risk-ozet [data-testid="stMetric"]      { padding: .04rem 0 .01rem 0; }

/* === Tablo/DataFrame (başlık + gövde aynı boy) === */
[data-testid="stDataFrame"] { font-size: .70rem; }

/* hem başlık hem gövdeyi eşitle + satır/padding’i daralt */
[data-testid="stDataFrame"] thead,
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td {
  font-size: .70rem;
  line-height: 1.15;
  padding-top: 4px;
  padding-bottom: 4px;
}

/* tablo araç çubuğu (indir vb.) bir tık küçült */
[data-testid="stElementToolbar"] button { transform: scale(.90); }
/* === Expander başlıkları === */
.st-expanderHeader, [data-baseweb="accordion"] { font-size: .80rem; }

/* === Radio/checkbox aralıklarını daralt === */
.stRadio > label, .stCheckbox > label { margin-bottom: .08rem; }

/* === Üst menü/footer (isteğe bağlı) === */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* === Özel KPI kartı (tooltip destekli) === */
.kpi{display:flex;flex-direction:column;gap:2px}
.kpi-label{font-size:.68rem;color:#6b7280}
.kpi-value{font-size:.95rem;font-weight:600}
</style>
"""

# ───────────────────────────── KPI satırı (tooltip'li, tek tip görünüm) ─────────────────────────────
def render_kpi_row(items: list[tuple[str, str | float, str]]):
    """
    items = [(label, value, tooltip), ...]
    Tooltip tarayıcı 'title' ile gösterilir.
    """
    cols = st.columns(len(items))
    for col, (label, value, tip) in zip(cols, items):
        col.markdown(
            f"""
            <div class="kpi" title="{tip}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- çeviri eşleştirmeleri ---
TR_LABEL = {
    "assault":   "Saldırı",
    "burglary":  "Konut/İşyeri Hırsızlığı",
    "theft":     "Hırsızlık",
    "robbery":   "Soygun",
    "vandalism": "Vandalizm",
}
CUE_MAP = {
    "assault":   ["bar/eğlence çıkışları", "meydan/park gözetimi"],
    "robbery":   ["metro/otobüs durağı & ATM", "dar sokak giriş/çıkış"],
    "theft":     ["otopark ve araç park alanları", "bagaj/bisiklet kilidi"],
    "burglary":  ["arka sokak & yükleme kapıları", "kapanış sonrası işyerleri"],
    "vandalism": ["okul/park/altgeçit", "inşaat sahası kontrolü"],
}

def actionable_cues(top_types: list[tuple[str, float]], max_items: int = 3) -> list[str]:
    tips: list[str] = []
    for crime, _ in top_types[:2]:
        tips.extend(CUE_MAP.get(crime, [])[:2])
    seen, out = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= max_items: break
    return out

def confidence_label(q10: float, q90: float) -> str:
    width = q90 - q10
    if width < 0.18: return "yüksek"
    if width < 0.30: return "orta"
    return "düşük"

def risk_window_text(start_iso: str, horizon_h: int) -> str:
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)
    if diurnal.size == 0:
        t2 = start
    else:
        thr = np.quantile(diurnal, 0.75)
        hot = np.where(diurnal >= thr)[0]
        if len(hot) == 0:
            t2 = start + timedelta(hours=horizon_h)
            return f"{start:%H:%M}–{t2:%H:%M}"
        splits = np.split(hot, np.where(np.diff(hot) != 1)[0] + 1)
        seg = max(splits, key=len)
        t1 = start + timedelta(hours=int(seg[0]))
        t2 = start + timedelta(hours=int(seg[-1]) + 1)
        t_peak = start + timedelta(hours=int(seg[len(seg)//2]))
        return f"{t1:%H:%M}–{t2:%H:%M} (tepe ≈ {t_peak:%H:%M})"
    return f"{start:%H:%M}–{t2:%H:%M}"

def render_result_card(df_agg: pd.DataFrame, geoid: str, start_iso: str, horizon_h: int):

# ───────────── Harita ─────────────
def color_for_tier(tier: str) -> str:
    return {"Yüksek": "#d62728", "Orta": "#ff7f0e", "Hafif": "#1f77b4"}.get(tier, "#1f77b4")

def build_map_fast(df_agg: pd.DataFrame, geo_features: list, geo_df: pd.DataFrame,
                   show_popups: bool = False, patrol: Dict | None = None) -> folium.Map:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty: return m

    color_map = {r[KEY_COL]: color_for_tier(r["tier"]) for _, r in df_agg.iterrows()}
    data_map = df_agg.set_index(KEY_COL).to_dict(orient="index")

    features = []
    for feat in geo_features:
        f = json.loads(json.dumps(feat))
        gid = f["properties"].get("id")
        row = data_map.get(gid)
        if row:
            expected = float(row["expected"])
            tier     = str(row["tier"])
            q10      = float(row["q10"]); q90 = float(row["q90"])
            types    = {t: float(row[t]) for t in CRIME_TYPES}
            top3     = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
            top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])
            f["properties"]["popup_html"] = (
                f"<b>{gid}</b><br/>E[olay] (ufuk): {expected:.2f} • Öncelik: <b>{tier}</b><br/>"
                f"<b>En olası 3 tip</b><ul style='margin-left:12px'>{top_html}</ul>"
                f"<i>Belirsizlik (saatlik ort.): q10={q10:.2f}, q90={q90:.2f}</i>"
            )
            f["properties"]["expected"] = round(expected, 2)
            f["properties"]["tier"]     = tier
        features.append(f)

    fc = {"type": "FeatureCollection", "features": features}
    def style_fn(feat):
        gid = feat["properties"].get("id")
        return {"fillColor": color_map.get(gid, "#9ecae1"), "color": "#666666", "weight": 0.3, "fillOpacity": 0.55}

    tooltip = folium.GeoJsonTooltip(
        fields=["id", "tier", "expected"],
        aliases=["GEOID", "Öncelik", "E[olay]"],
        localize=True, sticky=False
    ) if show_popups else None

    popup = folium.GeoJsonPopup(
        fields=["popup_html"],
        labels=False, parse_html=False, max_width=280
    ) if show_popups else None

    folium.GeoJson(fc, style_function=style_fn, tooltip=tooltip, popup=popup).add_to(m)

    # Üst %1 uyarı
    thr99 = np.quantile(df_agg["expected"].to_numpy(), 0.99)
    urgent = df_agg[df_agg["expected"] >= thr99].merge(
        geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL
    )
    for _, r in urgent.iterrows():
        folium.CircleMarker(
            location=[r["centroid_lat"], r["centroid_lon"]],
            radius=5, color="#000", fill=True, fill_color="#ff0000"
        ).add_to(m)

    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(html="<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'>"
                                         f" {z['id']} </div>")
            ).add_to(m)
    return m
