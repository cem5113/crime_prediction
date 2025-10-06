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

__all__ = [
    "SMALL_UI_CSS",
    "render_result_card",
    "build_map_fast",
    "render_kpi_row",
    "render_day_hour_heatmap",
]

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

# --- TR etiketleri ve saha ipuçları ---
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
        if len(out) >= max_items:
            break
    return out

def confidence_label(q10: float, q90: float) -> str:
    width = q90 - q10
    if width < 0.18:
        return "yüksek"
    if width < 0.30:
        return "orta"
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

# ───────────────────────────── SONUÇ KARTI ─────────────────────────────
def render_result_card(df_agg: pd.DataFrame, geoid: str, start_iso: str, horizon_h: int):
    if df_agg is None or df_agg.empty or geoid is None:
        st.info("Bölge seçilmedi.")
        return

    row = df_agg.loc[df_agg[KEY_COL] == geoid]
    if row.empty:
        st.info("Seçilen bölge için veri yok.")
        return
    row = row.iloc[0].to_dict()

    # Near-repeat (varsa)
    nr = float(row.get("nr_boost", 0.0))

    # Tür bazında λ ve P(≥1)
    type_lams = {t: float(row.get(t, 0.0)) for t in CRIME_TYPES}
    type_probs = {TR_LABEL.get(t, t): 1.0 - math.exp(-lam) for t, lam in type_lams.items()}
    probs_sorted = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)

    # İlk iki tür için 90% PI metni
    pi90_lines: list[str] = []
    for name_tr, _p in probs_sorted[:2]:
        t_eng = next((k for k, v in TR_LABEL.items() if v == name_tr), None)
        if t_eng is None:
            continue
        lam = type_lams.get(t_eng, 0.0)
        lo, hi = pois_pi90(lam)
        pi90_lines.append(f"{name_tr}: {lam:.1f} ({lo}–{hi})")

    # Güven ve zaman penceresi
    q10 = float(row.get("q10", 0.0))
    q90 = float(row.get("q90", 0.0))
    conf_txt = confidence_label(q10, q90)
    win_text = risk_window_text(start_iso, horizon_h)

    # UI
    st.markdown("### 🧭 Sonuç Kartı")
    c1, c2, c3 = st.columns([1.0, 1.2, 1.2])

    with c1:
        st.metric("Bölge (GEOID)", geoid)
        st.metric("Öncelik", str(row.get("tier", "—")))
        st.metric("Ufuk", f"{horizon_h} saat")

    with c2:
        st.markdown("**En olası suç türleri (P≥1)**")
        for name_tr, p in probs_sorted[:5]:
            st.write(f"- {name_tr}: {p:.2f}")

    with c3:
        st.markdown("**Beklenen sayılar (90% PI)**")
        for line in pi90_lines:
            st.write(f"- {line}")

    st.markdown("---")

    # Top-2 öneri metni
    top2 = [name for name, _ in probs_sorted[:2]]
    st.markdown(f"**Top-2 öneri:** {', '.join(top2) if top2 else '—'}")

    # Kolluğa pratik öneriler
    try:
        top_types_eng = []
        for name_tr, _ in probs_sorted[:2]:
            t_eng = next((k for k, v in TR_LABEL.items() if v == name_tr), None)
            if t_eng:
                top_types_eng.append((t_eng, type_lams.get(t_eng, 0.0)))
        cues = actionable_cues(top_types_eng, max_items=3)
    except Exception:
        cues = []

    # Near-repeat satırı (tek yerde)
    if nr > 0:
        st.markdown(
            f"- **Near-repeat etkisi:** {nr:.2f} (0=etki yok, 1=yüksek). "
            "Taze olay çevresinde kısa ufukta risk artar."
        )

    st.markdown(f"- **Risk penceresi:** {win_text}")
    st.markdown(f"- **Güven:** {conf_txt} (q10={q10:.2f}, q90={q90:.2f})")

    if cues:
        st.markdown("**Kolluğa öneriler:**")
        for c in cues:
            st.write(f"- {c}")

# ───────────── Harita ─────────────
def color_for_tier(tier: str) -> str:
    return {"Yüksek": "#d62728", "Orta": "#ff7f0e", "Hafif": "#1f77b4"}.get(tier, "#1f77b4")

def build_map_fast(
    df_agg: pd.DataFrame,
    geo_features: list,
    geo_df: pd.DataFrame,
    show_popups: bool = False,
    patrol: Dict | None = None,
    *,
    show_poi: bool = False,
    show_transit: bool = False,
) -> folium.Map:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty:
        return m

    # --- hücre stilleri & popup
    color_map = {r[KEY_COL]: color_for_tier(r["tier"]) for _, r in df_agg.iterrows()}
    data_map = df_agg.set_index(KEY_COL).to_dict(orient="index")

    features = []
    for feat in geo_features:
        f = json.loads(json.dumps(feat))  # derin kopya
        gid = f["properties"].get("id")
        row = data_map.get(gid)
        if row:
            expected = float(row["expected"])
            tier     = str(row["tier"])
            q10      = float(row.get("q10", 0.0))
            q90      = float(row.get("q90", 0.0))
            types    = {t: float(row.get(t, 0.0)) for t in CRIME_TYPES}
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

    tooltip = (folium.GeoJsonTooltip(
        fields=["id", "tier", "expected"],
        aliases=["GEOID", "Öncelik", "E[olay]"],
        localize=True, sticky=False
    ) if show_popups else None)

    popup = (folium.GeoJsonPopup(
        fields=["popup_html"], labels=False, parse_html=False, max_width=280
    ) if show_popups else None)

    folium.GeoJson(fc, style_function=style_fn, tooltip=tooltip, popup=popup).add_to(m)

    # ---------- POI / Transit overlay'leri ----------
    def _read_first_existing_csv(paths: list[str]) -> pd.DataFrame | None:
        for p in paths:
            try:
                return pd.read_csv(p)
            except Exception:
                continue
        return None

    if show_poi:
        try:
            poi_df = _read_first_existing_csv([
                "data/sf_pois_cleaned_with_geoid.csv",
                "data/poi.csv",
            ])
            if poi_df is not None and not poi_df.empty:
                lat_col = "latitude" if "latitude" in poi_df.columns else ("lat" if "lat" in poi_df.columns else None)
                lon_col = "longitude" if "longitude" in poi_df.columns else ("lon" if "lon" in poi_df.columns else None)
                if lat_col and lon_col:
                    fg_poi = folium.FeatureGroup(name="POI", show=True)
                    for _, r in poi_df.head(2000).iterrows():  # performans sınırı
                        folium.CircleMarker(
                            location=[float(r[lat_col]), float(r[lon_col])],
                            radius=2, color="#3b82f6", fill=True, fill_color="#3b82f6", fill_opacity=0.6, opacity=0.7
                        ).add_to(fg_poi)
                    fg_poi.add_to(m)
        except Exception:
            pass

    if show_transit:
        try:
            bus_df = _read_first_existing_csv([
                "data/sf_bus_stops_with_geoid.csv",
                "data/sf_bus_stops.csv",
                "data/transit_bus_stops.csv",
            ])
            train_df = _read_first_existing_csv([
                "data/sf_train_stops_with_geoid.csv",
                "data/sf_train_stops.csv",
                "data/transit_train_stops.csv",
            ])
            fg_tr = folium.FeatureGroup(name="Transit", show=True)

            if bus_df is not None and not bus_df.empty:
                blat = "latitude" if "latitude" in bus_df.columns else ("lat" if "lat" in bus_df.columns else None)
                blon = "longitude" if "longitude" in bus_df.columns else ("lon" if "lon" in bus_df.columns else None)
                if blat and blon:
                    for _, r in bus_df.head(2000).iterrows():
                        folium.CircleMarker(
                            location=[float(r[blat]), float(r[blon])],
                            radius=1.6, color="#10b981", fill=True, fill_color="#10b981", fill_opacity=0.55, opacity=0.6
                        ).add_to(fg_tr)

            if train_df is not None and not train_df.empty:
                tlat = "latitude" if "latitude" in train_df.columns else ("lat" if "lat" in train_df.columns else None)
                tlon = "longitude" if "longitude" in train_df.columns else ("lon" if "lon" in train_df.columns else None)
                if tlat and tlon:
                    for _, r in train_df.head(1500).iterrows():
                        folium.CircleMarker(
                            location=[float(r[tlat]), float(r[tlon])],
                            radius=2.2, color="#ef4444", fill=True, fill_color="#ef4444", fill_opacity=0.6, opacity=0.75
                        ).add_to(fg_tr)

            if len(getattr(fg_tr, "_children", {})) > 0:
                fg_tr.add_to(m)
        except Exception:
            pass

    try:
        folium.LayerControl(collapsed=True).add_to(m)
    except Exception:
        pass

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

    # Devriye rotaları
    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(html="<div style='background:#111;color:#fff;padding:2px 6px;border-radius:6px'>"
                                         f" {z['id']} </div>")
            ).add_to(m)

    return m


# ───────────── Gün × Saat Isı Matrisi ─────────────
def render_day_hour_heatmap(agg: pd.DataFrame, start_iso: str | None = None, horizon_h: int | None = None):
    """
    - agg 'dow' ve 'hour' içeriyorsa direkt pivot.
    - Değilse ve start_iso & horizon_h verilmişse, toplam expected’ı diurnal profile göre saatlere dağıtır.
    - Aksi halde 24 saatlik varsayılan bir dağıtım üretir.
    """
    if agg is None or agg.empty:
        st.caption("Isı matrisi için veri yok.")
        return

    # 1) Hazır kolonlarla pivot
    if {"dow", "hour"}.issubset(agg.columns):
        mat = (
            agg.pivot_table(index="dow", columns="hour", values="expected", aggfunc="sum")
               .reindex(index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fill_value=0.0)
               .reindex(columns=list(range(24)), fill_value=0.0)
        )
        st.dataframe(mat.round(2), use_container_width=True)
        return

    # 2) start/horizon ile sentez
    if start_iso is not None and horizon_h is not None:
        start = pd.to_datetime(start_iso)
        hours = np.arange(int(horizon_h))
        diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)
        w = diurnal / (diurnal.sum() + 1e-12)
        total_expected = float(agg["expected"].sum())

        dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        mat = pd.DataFrame(0.0, index=dow_labels, columns=[f"{h:02d}" for h in range(24)])
        for h, weight in enumerate(w):
            dt = start + timedelta(hours=int(h))
            mat.loc[dow_labels[dt.dayofweek], f"{dt.hour:02d}"] += total_expected * float(weight)

        st.dataframe(mat.round(2), use_container_width=True)
        return

    # 3) Fallback: 24 saat varsayılan
    start = pd.Timestamp.utcnow()
    hours = np.arange(24)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)
    w = diurnal / (diurnal.sum() + 1e-12)
    total_expected = float(agg["expected"].sum())

    dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    mat = pd.DataFrame(0.0, index=dow_labels, columns=[f"{h:02d}" for h in range(24)])
    for h, weight in enumerate(w):
        dt = start + timedelta(hours=int(h))
        mat.loc[dow_labels[dt.dayofweek], f"{dt.hour:02d}"] += total_expected * float(weight)

    st.dataframe(mat.round(2), use_container_width=True)
