# utils/ui.py
from __future__ import annotations
import math, json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import folium
import streamlit as st
from zoneinfo import ZoneInfo

from utils.constants import KEY_COL, CRIME_TYPES
from utils.forecast import pois_pi90
# ────────────────────────── CSS (TÜM yazılar küçük, sadece H1 büyük) ─────────────────────────
SMALL_UI_CSS = """
<style>
/* GENEL: daha küçük font */
html, body, [class*="css"] { font-size: 14px; }

/* Başlıklar */
h1 { font-size: 1.9rem; line-height: 1.25; margin: .6rem 0 .6rem 0; } /* Uygulama adı BÜYÜK kalsın */
h2 { font-size: 1.10rem; margin: .35rem 0; }
h3 { font-size: 0.98rem;  margin: .30rem 0; }

/* İç boşluklar */
section.main > div.block-container { padding-top: 1.0rem; padding-bottom: .25rem; }
[data-testid="stSidebar"] .block-container { padding-top: .5rem; padding-bottom: .5rem; }

/* Metric kartları (GENEL küçük) */
[data-testid="stMetricValue"] { font-size: 1.15rem; }
[data-testid="stMetricLabel"] { font-size: .78rem; color: #666; }

/* Dataframe yazısı */
[data-testid="stDataFrame"] { font-size: .84rem; }

/* Form etiketleri */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[role="radiogroup"] label { font-size: .88rem; }

/* Expander başlığı */
.st-expanderHeader, [data-baseweb="accordion"] { font-size: .88rem; }

/* Sadece Risk Özeti bloğunu daha da küçült */
#risk-ozet [data-testid="stMetricValue"] { font-size: 1.05rem; line-height: 1.1; }
#risk-ozet [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #6b7280; }
#risk-ozet [data-testid="stMetric"]      { padding: 0.15rem 0 0.1rem 0; }

/* (opsiyonel) üst menü/footer gizle */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""

def show_last_update_badge(last_updated_utc):
    if not last_updated_utc:
        return
    try:
        sf = ZoneInfo("America/Los_Angeles")
        dt_sf = last_updated_utc.astimezone(sf)
        ts_txt = dt_sf.strftime("%d-%m-%Y %H:%M")
    except Exception:
        ts_txt = last_updated_utc.strftime("%d-%m-%Y %H:%M")
    st.markdown("""
    <style>
      .update-badge {
        position: absolute; right: 0; top: -6px;
        background: #eef2ff; color: #1e293b;
        border: 1px solid #c7d2fe; border-radius: 9999px;
        padding: 6px 10px; font-size: 13px;
      }
      .header-row { position: relative; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='update-badge'>🕒 Son veriler: {ts_txt}</div>", unsafe_allow_html=True)

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
    if df_agg is None or df_agg.empty or geoid is None:
        st.info("Bölge seçilmedi.")
        return
    row = df_agg.loc[df_agg[KEY_COL] == geoid]
    if row.empty:
        st.info("Seçilen bölge için veri yok.")
        return
    row = row.iloc[0].to_dict()

    type_lams = {t: float(row.get(t, 0.0)) for t in CRIME_TYPES}
    type_probs = {TR_LABEL[t]: 1.0 - math.exp(-lam) for t, lam in type_lams.items()}
    probs_sorted = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)

    top2 = [name for name, _ in probs_sorted[:2]]
    pi90_lines = []
    for name_tr, _p in probs_sorted[:2]:
        t_eng = next(k for k, v in TR_LABEL.items() if v == name_tr)
        lam = type_lams[t_eng]
        lo, hi = pois_pi90(lam)
        pi90_lines.append(f"{name_tr}: {lam:.1f} ({lo}–{hi})")

    q10 = float(row.get("q10", 0.0)); q90 = float(row.get("q90", 0.0))
    conf_txt = confidence_label(q10, q90)
    win_text = risk_window_text(start_iso, horizon_h)

    st.markdown("### 🧭 Sonuç Kartı")
    c1, c2, c3 = st.columns([1.0, 1.2, 1.2])
    with c1:
        st.metric("Bölge (GEOID)", geoid)
        st.metric("Öncelik", str(row.get("tier", "—")))
        st.metric("Ufuk", f"{horizon_h} saat")
    with c2:
        st.markdown("**Olasılıklar (P≥1, tür bazında)**")
        for name_tr, p in probs_sorted:
            st.write(f"- {name_tr}: {p:.2f}")
    with c3:
        st.markdown("**Beklenen sayılar (90% PI)**")
        for line in pi90_lines:
            st.write(f"- {line}")
    st.markdown("---")
    st.markdown(f"**Top-2 öneri:** {', '.join(top2) if top2 else '—'}")
    st.markdown(f"- **Risk penceresi:** {win_text}  \n- **Güven:** {conf_txt} (q10={q10:.2f}, q90={q90:.2f})")

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

    tooltip = folium.GeoJsonTooltip(fields=["id", "tier", "expected"],
                                    aliases=["GEOID", "Öncelik", "E[olay]"],
                                    localize=True, sticky=False) if show_popups else None
    popup = folium.GeoJsonPopup(fields=["popup_html"], labels=False, parse_html=False, max_width=280) if show_popups else None
    folium.GeoJson(fc, style_function=style_fn, tooltip=tooltip, popup=popup).add_to(m)

    # Üst %1 uyarı
    thr99 = np.quantile(df_agg["expected"].to_numpy(), 0.99)
    urgent = df_agg[df_agg["expected"] >= thr99].merge(
        geo_df[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL
    )
    for _, r in urgent.iterrows():
        folium.CircleMarker(location=[r["centroid_lat"], r["centroid_lon"]],
                            radius=5, color="#000", fill=True, fill_color="#ff0000").add_to(m)

    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker([z["centroid"]["lat"], z["centroid"]["lon"]],
                          icon=folium.DivIcon(html="<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'>"
                                                    f" {z['id']} </div>")).add_to(m)
    return m
