from __future__ import annotations
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo  # yerel saat rozetinde kullanacağız

import streamlit as st
import altair as alt

# --- SAYFA AYARLARI (Streamlit'te İLK çağrı olmalı!) ---
st.set_page_config(page_title="SUTAM: Suç Tahmin Modeli", layout="wide")

# --- “Son güncelleme” rozetini çizen yardımcı ---
def show_last_update_badge(last_updated_utc: datetime | None) -> None:
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

    # başlığın bulunduğu satırın sağ üstüne otursun
    st.markdown(f"<div class='update-badge'>🕒 Son veriler: {ts_txt}</div>", unsafe_allow_html=True)

# --- last_update.json güvenli okuma ---
def read_last_update(path: str = "last_update.json") -> datetime | None:
    p = Path(os.environ.get("LAST_UPDATE_JSON", path))
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            ts_str = json.load(f).get("last_update_utc")
        return datetime.fromisoformat(ts_str) if ts_str else None
    except Exception:
        return None

last_updated_utc = read_last_update()

# --- Stil (başlık 1 tık büyük + 1 tık aşağı) ---
SMALL_UI_CSS = """
<style>
/* Genel yazı boyutu */
html, body, [class*="css"] { font-size: 14px; }

/* Başlıklar */
h1 { font-size: 1.9rem; line-height: 1.25; margin: .6rem 0 .6rem 0; } /* önce 1.6rem ve .1rem idi */
h2 { font-size: 1.15rem; margin: .4rem 0; }
h3 { font-size: 1.00rem; margin: .3rem 0; }

/* Ana içerik ve sidebar iç boşlukları */
section.main > div.block-container { padding-top: 1.1rem; padding-bottom: .25rem; } /* önce .5rem idi */
[data-testid="stSidebar"] .block-container { padding-top: .6rem; padding-bottom: .6rem; }

/* Metric kartları */
[data-testid="stMetricValue"] { font-size: 1.25rem; }
[data-testid="stMetricLabel"] { font-size: .80rem; color: #666; }

/* Dataframe yazısı */
[data-testid="stDataFrame"] { font-size: .85rem; }

/* Giriş bileşenleri etiketleri */
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[role="radiogroup"] label { font-size: .9rem; }

/* Expander başlığı */
.st-expanderHeader, [data-baseweb="accordion"] { font-size: .9rem; }

/* (opsiyonel) üst menü / footer gizle */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
"""
st.markdown(SMALL_UI_CSS, unsafe_allow_html=True)

# Başlık (st.title bırak; CSS bunu büyütüp aşağı alacak)
st.title("SUTAM: Suç Tahmin Modeli")


# =============================
# SABİTLER
# =============================
SF_TZ_OFFSET = -7  # PDT kabaca; prod'da pytz/pytzdata kullanın
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]
KEY_COL = "geoid"
CACHE_VERSION = "v2-geo-poisson"

rng = np.random.default_rng(42)

# =============================
# YARDIMCI FONKSİYONLAR
# =============================

def now_sf_iso() -> str:
    return (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).isoformat(timespec="seconds")

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
        return pd.DataFrame(columns=[key_field, "centroid_lon", "centroid_lat"]), []
    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geoid = str(
            props.get(key_field)
            or props.get(key_field.upper())
            or props.get("GEOID")
            or props.get("geoid")
            or ""
        ).strip()
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

def nearest_geoid(lat: float, lon: float) -> str | None:
    if GEO_DF.empty:
        return None
    la = GEO_DF["centroid_lat"].to_numpy()
    lo = GEO_DF["centroid_lon"].to_numpy()
    d2 = (la - lat) ** 2 + (lo - lon) ** 2
    i = int(np.argmin(d2))
    return str(GEO_DF.iloc[i][KEY_COL])

def _extract_latlon_from_ret(ret) -> Tuple[float, float] | None:
    if not ret:
        return None
    lc = ret.get("last_clicked")
    if lc is None:
        return None
    # list/tuple: [lat, lon]
    if isinstance(lc, (list, tuple)) and len(lc) >= 2:
        return float(lc[0]), float(lc[1])
    # dict: {"lat":..,"lng":..} veya {"lat":..,"lon":..} veya {"latlng": {...}}
    if isinstance(lc, dict):
        if "lat" in lc and ("lng" in lc or "lon" in lc):
            return float(lc["lat"]), float(lc.get("lng", lc.get("lon")))
        ll = lc.get("latlng")
        if isinstance(ll, (list, tuple)) and len(ll) >= 2:
            return float(ll[0]), float(ll[1])
        if isinstance(ll, dict) and "lat" in ll and ("lng" in ll or "lon" in ll):
            return float(ll["lat"]), float(ll.get("lng", ll.get("lon")))
    return None

def resolve_clicked_gid(ret: dict) -> tuple[str | None, tuple[float, float] | None]:
    """st_folium dönüşünden güvenli şekilde (gid, latlon) çıkarır."""
    gid, latlon = None, None
    obj = ret.get("last_object_clicked") if isinstance(ret, dict) else None

    # 1) GeoJSON / Feature -> id alanları
    if isinstance(obj, dict):
        props = obj.get("properties", {}) or obj.get("feature", {}).get("properties", {}) or {}
        gid = str(
            obj.get("id")
            or props.get("id")
            or props.get(KEY_COL)
            or props.get("GEOID")
            or ""
        ).strip() or None

        # 2) Nesneden koordinat yakalamaya çalış (Point, CircleMarker vb.)
        if not latlon:
            # GeoJSON Point
            geom = obj.get("geometry") or obj.get("feature", {}).get("geometry")
            if isinstance(geom, dict) and geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    latlon = (float(coords[1]), float(coords[0]))
            # Bazı katmanlar lat/lng saklar
            if not latlon:
                lat = obj.get("lat") or obj.get("latlng", {}).get("lat") or obj.get("location", {}).get("lat")
                lng = obj.get("lng") or obj.get("latlng", {}).get("lng") or obj.get("location", {}).get("lng") or obj.get("lon")
                if lat is not None and lng is not None:
                    latlon = (float(lat), float(lng))

    # 3) Haritaya tık (poligon yerine zemin): last_clicked
    if not gid:
        if not latlon:
            lc = ret.get("last_clicked") if isinstance(ret, dict) else None
            if isinstance(lc, (list, tuple)) and len(lc) >= 2:
                latlon = (float(lc[0]), float(lc[1]))
            elif isinstance(lc, dict) and "lat" in lc and ("lng" in lc or "lon" in lc):
                latlon = (float(lc["lat"]), float(lc.get("lng", lc.get("lon"))))
        if latlon:
            gid = nearest_geoid(latlon[0], latlon[1])

    return gid, latlon

def local_explain(df_agg: pd.DataFrame, geoid: str, start_iso: str, horizon_h: int) -> Dict:
    # Hücre satırı
    row = df_agg.loc[df_agg[KEY_COL] == geoid]
    if row.empty:
        return {}
    row = row.iloc[0]

    # Mekansal bileşen
    ixs = GEO_DF.index[GEO_DF[KEY_COL] == geoid]
    if len(ixs) == 0:
        return {}
    i = int(ixs[0])
    spatial = float(BASE_INT[i])

    # Zaman bileşeni (aynısını aggregate_fast’ta kullanıyoruz)
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)
    temporal = float(np.mean(diurnal))

    # Senaryo/ek faktör (şimdilik 1.0; ileride hava/etkinlik ekleyince değiştir)
    scenario = 1.0

    # Katkıları "E[olay]" büyüklüğüne oranlayarak normalize et
    expected = float(row["expected"])
    parts_raw = {"Mekânsal sıcak-nokta": spatial, "Saat etkisi": temporal, "Senaryo": scenario}
    s = sum(parts_raw.values())
    contribs = {k: (v / s) * expected for k, v in parts_raw.items()}

    # Top 3 suç
    top_types = sorted([(t, float(row[t])) for t in CRIME_TYPES], key=lambda x: x[1], reverse=True)[:3]

    return {
        "expected": expected,
        "tier": str(row["tier"]),
        "q10": float(row["q10"]),
        "q90": float(row["q90"]),
        "contribs": contribs,
        "top_types": top_types,
    }

def _fmt_hhmm(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def risk_window_text(start_iso: str, horizon_h: int) -> str:
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)
    if diurnal.size == 0:
        return f"{_fmt_hhmm(start)}–{_fmt_hhmm(start)}"
    thr = np.quantile(diurnal, 0.75)
    hot = np.where(diurnal >= thr)[0]
    if len(hot) == 0:
        t2 = start + timedelta(hours=horizon_h)
        return f"{_fmt_hhmm(start)}–{_fmt_hhmm(t2)}"
    # en uzun ardışık dilimi seç
    splits = np.split(hot, np.where(np.diff(hot) != 1)[0] + 1)
    seg = max(splits, key=len)
    t1 = start + timedelta(hours=int(seg[0]))
    t2 = start + timedelta(hours=int(seg[-1]) + 1)
    t_peak = start + timedelta(hours=int(seg[len(seg)//2]))
    return f"{_fmt_hhmm(t1)}–{_fmt_hhmm(t2)} (tepe ≈ {_fmt_hhmm(t_peak)})"

def confidence_label(q10: float, q90: float) -> str:
    width = q90 - q10
    if width < 0.18: return "yüksek"
    if width < 0.30: return "orta"
    return "düşük"

CUE_MAP = {
    "assault":   ["bar/eğlence çıkışları", "meydan/park gözetimi"],
    "robbery":   ["metro/otobüs durağı & ATM çevresi", "dar sokak giriş-çıkışları"],
    "theft":     ["otopark/araç park alanları", "bagaj/bisiklet kilit kontrolü"],
    "burglary":  ["arka sokaklar & yükleme kapıları", "kapanış sonrası işyerleri"],
    "vandalism": ["okul/park/altgeçit çevresi", "inşaat sahası kontrolü"],
}

def actionable_cues(top_types: list[tuple[str, float]], max_items: int = 3) -> list[str]:
    tips: list[str] = []
    for crime, _ in top_types[:2]:
        tips.extend(CUE_MAP.get(crime, [])[:2])
    # yinelenenleri at, ilk max_items’i al
    seen, out = set(), []
    for t in tips:
        if t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= max_items: break
    return out

# --------- HIZLI AGGREGATION BLOĞU (EKLE: GEO_DF satırının hemen altına) ---------
@st.cache_data(show_spinner=False)
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()
    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    noise = 0.07  # küçük sabit gürültü
    return 0.2 + 0.8 * (peak1 + peak2) + noise

BASE_INT = precompute_base_intensity(GEO_DF)

def p_to_lambda_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 0.999999)
    return -np.log1p(-p)

@st.cache_data(show_spinner=False)
def aggregate_fast(start_iso: str, horizon_h: int) -> pd.DataFrame:
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    # tüm hücreler × saatler
    p = np.clip(BASE_INT[:, None] * diurnal[None, :], 0, 1)
    p_any = np.clip(0.05 + 0.5 * p, 0, 0.98)

    lam = p_to_lambda_array(p_any)
    expected = lam.sum(axis=1)
    q10 = np.maximum(0.0, p_any - 0.08).mean(axis=1)
    q90 = np.minimum(1.0, p_any + 0.08).mean(axis=1)

    alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
    W = rng.dirichlet(alpha, size=len(GEO_DF))
    types = expected[:, None] * W
    assault, burglary, theft, robbery, vandalism = types.T

    out = pd.DataFrame({
        KEY_COL: GEO_DF[KEY_COL].to_numpy(),
        "expected": expected,
        "q10": q10, "q90": q90,
        "assault": assault, "burglary": burglary, "theft": theft,
        "robbery": robbery, "vandalism": vandalism,
    })

    q90_thr = out["expected"].quantile(0.90)
    q70_thr = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90_thr, out["expected"] >= q70_thr],
        ["Yüksek", "Orta"], default="Hafif",
    )
    return out

# --- Sentetik yoğunluk (sadece prototip) ---
def scenario_multipliers(_scenario: Dict) -> float:
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
    agg_map = {"p_any": "mean", "q10": "mean", "q90": "mean"} | {t: "mean" for t in CRIME_TYPES}
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

# --- Poisson yardımcıları ---
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

# === Sonuç Kartı yardımcıları (EKLE: prob_ge_k biter bitmez) ===

TR_LABEL = {
    "assault":   "Saldırı",
    "burglary":  "Konut/İşyeri Hırsızlığı",
    "theft":     "Hırsızlık",
    "robbery":   "Soygun",
    "vandalism": "Vandalizm",
}

def pois_quantile(lam: float, q: float) -> int:
    """Poisson q-kantili (scipy'siz basit hesap)."""
    k = 0
    while pois_cdf(k, lam) < q and k < 10_000:
        k += 1
    return k

def pois_pi90(lam: float) -> tuple[int, int]:
    """Poisson için yaklaşık %90 PI (5.–95. yüzdelikler)."""
    lo = pois_quantile(lam, 0.05)
    hi = pois_quantile(lam, 0.95)
    return lo, hi

def render_result_card(df_agg: pd.DataFrame, geoid: str, start_iso: str, horizon_h: int):
    """Harita tıklaması/selection sonrası 'Sonuç Kartı'nı çizer."""
    if df_agg is None or df_agg.empty or geoid is None:
        st.info("Bölge seçilmedi.")
        return
    row = df_agg.loc[df_agg[KEY_COL] == geoid]
    if row.empty:
        st.info("Seçilen bölge için veri yok.")
        return
    row = row.iloc[0].to_dict()

    # Tür bazında λ ve P(≥1)
    type_lams = {t: float(row.get(t, 0.0)) for t in CRIME_TYPES}
    type_probs = {TR_LABEL[t]: 1.0 - math.exp(-lam) for t, lam in type_lams.items()}
    probs_sorted = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)

    # Top-2 öneri
    top2 = [name for name, _ in probs_sorted[:2]]

    # Tür bazlı 90% PI (Poisson)
    pi90_lines = []
    for name_tr, _p in probs_sorted[:2]:
        t_eng = next(k for k, v in TR_LABEL.items() if v == name_tr)
        lam = type_lams[t_eng]
        lo, hi = pois_pi90(lam)
        pi90_lines.append(f"{name_tr}: {lam:.1f} ({lo}–{hi})")

    # Genel açıklama
    ex = local_explain(df_agg, geoid, start_iso, horizon_h) or {}
    win_text = risk_window_text(start_iso, horizon_h)
    conf_txt = confidence_label(ex.get("q10", 0.0), ex.get("q90", 0.0))
    tips = actionable_cues(ex.get("top_types", []))
    drivers = ex.get("contribs", {})
    if drivers:
        total = sum(drivers.values()) or 1.0
        drv_txt = ", ".join([f"{k} %{round(100*v/total)}" for k, v in drivers.items()])
    else:
        drv_txt = "—"

    # === KART UI ===
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
    st.markdown(
        f"- **Risk penceresi:** {win_text}  \n"
        f"- **Sürücüler:** {drv_txt}  \n"
        f"- **Güven:** {conf_txt} (q10={ex.get('q10', 0):.2f}, q90={ex.get('q90', 0):.2f})"
    )
    if tips:
        st.markdown("**Eylem önerileri:**")
        for t in tips[:3]:
            st.markdown(f"- {t}")

# --- Görünüm için agregasyon ---
def aggregate_for_view(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # saatlik/periodik p_any → λ
    df["lam"] = p_to_lambda(df["p_any"])

    # saatlik belirsizliklerin ortalaması (popup için)
    mean_part = df.groupby(KEY_COL, as_index=False).agg({"q10": "mean", "q90": "mean"})
    # ufuk boyunca beklenen toplam olay (λ toplamı)
    sum_part  = df.groupby(KEY_COL, as_index=False).agg({"lam": "sum"} | {t: "sum" for t in CRIME_TYPES})

    out = mean_part.merge(sum_part, on=KEY_COL).rename(columns={"lam": "expected"})

    # Öncelik sınıfları (quantile) – expected üzerinden
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

    lam = tab["expected"].to_numpy()
    tab["P(≥1)%"] = [round(prob_ge_k(l, 1) * 100, 1) for l in lam]
    tab["P(≥2)%"] = [round(prob_ge_k(l, 2) * 100, 1) for l in lam]
    tab["P(≥3)%"] = [round(prob_ge_k(l, 3) * 100, 1) for l in lam]

    tab["expected"] = tab["expected"].round(2)
    for t in CRIME_TYPES:
        tab[t] = tab[t].round(3)

    return tab.rename(columns={"expected": "E[olay] (λ)"})

# =============================
# RAPOR METRİK YARDIMCILARI
# =============================
@st.cache_data(show_spinner=False)
def load_events(path: str = "data/events.csv") -> pd.DataFrame:
    """Gerçek olaylar. Sütunlar: ts (ISO/datetime), geoid (str), type (ops.)."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ts", KEY_COL, "type"])
    df = pd.read_csv(p, parse_dates=["ts"])
    df[KEY_COL] = df[KEY_COL].astype(str)
    return df

def slice_events(events: pd.DataFrame, start_iso: str, horizon_h: int) -> pd.DataFrame:
    start = datetime.fromisoformat(start_iso)
    end = start + timedelta(hours=horizon_h)
    m = (events["ts"] >= start) & (events["ts"] < end)
    return events.loc[m, [KEY_COL, "ts", "type"]].copy()

def prep_eval_frames(agg: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    cnts = ev.groupby(KEY_COL).size().rename("y_count")
    ybin = (cnts > 0).astype(int).rename("y")
    out = (
        agg[[KEY_COL, "expected"]]
        .merge(ybin, on=KEY_COL, how="left")
        .merge(cnts, on=KEY_COL, how="left")
        .fillna({"y": 0, "y_count": 0})
    )
    out["p"] = 1.0 - np.exp(-out["expected"].clip(lower=0))
    return out

def pai_at_k(eval_df: pd.DataFrame, k_pct: int) -> tuple[float, float]:
    """PAI@K ve alan içindeki olay payını döndürür."""
    if eval_df["y_count"].sum() == 0:
        return float("nan"), 0.0
    frac = k_pct / 100.0
    n = len(eval_df)
    top_k = max(1, int(round(n * frac)))
    d = eval_df.sort_values("expected", ascending=False).head(top_k)
    event_share = d["y_count"].sum() / eval_df["y_count"].sum()
    return (event_share / frac), event_share

# ---- Küçük HTML üreticileri (indirme için) ----
def brief_html(top5: pd.DataFrame, lines: list[str], patrol_rows: list[dict], start_iso: str, horizon_h: int) -> str:
    win = risk_window_text(start_iso, horizon_h)
    rows_html = "".join(
        f"<tr><td>{r[KEY_COL]}</td><td>{r['E[olay] (λ)']}</td></tr>" for _, r in top5[[KEY_COL,'E[olay] (λ)']].iterrows()
    )
    anons_html = "".join(f"<li>{ln}</li>" for ln in lines)
    patrol_html = "".join(
        f"<tr><td>{z['zone']}</td><td>{z['cells_planned']}/{z['capacity_cells']}</td>"
        f"<td>{z['eta_minutes']} dk</td><td>{z['avg_risk(E[olay])']}</td></tr>" for z in patrol_rows
    )
    return f"""
    <html><head><meta charset="utf-8"><style>
    body{{font-family:system-ui,Arial;}} h2{{margin:6px 0}} table{{border-collapse:collapse;width:100%}}
    td,th{{border:1px solid #ccc;padding:6px;text-align:left}} small{{color:#555}}
    </style></head><body>
    <h2>Vardiya Brifi</h2>
    <small>Risk penceresi: {win}</small>
    <h3>Top-5 Bölge</h3>
    <table><tr><th>GEOID</th><th>E[olay] (λ)</th></tr>{rows_html}</table>
    <h3>Radyo Anonsları</h3><ul>{anons_html}</ul>
    <h3>Önerilen Devriye</h3>
    <table><tr><th>Zone</th><th>Plan/Kapasite</th><th>Süre</th><th>Ort. Risk</th></tr>{patrol_html}</table>
    <p><small>Not: Güven seviyesi metin içinde hücre bazında yer alır.</small></p>
    </body></html>
    """

def weekly_html(pai10: float, share10: float, top10: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><td>{r[KEY_COL]}</td><td>{r['E[olay] (λ)']}</td></tr>" for _, r in top10[[KEY_COL,'E[olay] (λ)']].iterrows()
    )
    pai_txt = "—" if np.isnan(pai10) else f"{pai10:.2f} (alan %10 → olay %{share10*100:.1f})"
    return f"""
    <html><head><meta charset="utf-8"><style>
    body{{font-family:system-ui,Arial;}} table{{border-collapse:collapse;width:100%}}
    td,th{{border:1px solid #ccc;padding:6px;text-align:left}}
    </style></head><body>
    <h2>Haftalık Operasyon Özeti</h2>
    <p><b>PAI@10:</b> {pai_txt}</p>
    <h3>Top-10 Hotspot</h3>
    <table><tr><th>GEOID</th><th>E[olay] (λ)</th></tr>{rows}</table>
    </body></html>
    """

# --- Devriye kümeleme ---
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
                centroids[c] = coords[idx_sorted[0]]
            else:
                w = weights[m][:, None]
                centroids[c] = (coords[m] * w).sum(axis=0) / max(1e-6, w.sum())
    return centroids, assign

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# YENİ: K ve görev süresiyle devriye planlama
def allocate_patrols(
    df_agg: pd.DataFrame,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
) -> Dict:
    """
    k_planned: planlanan ekip sayısı
    duty_minutes: her ekibin sahada geçireceği süre (dk)
    cell_minutes: 1 hücrede ort. kontrol süresi (dk)
    travel_overhead: % seyir/aktarma payı (0.40 = %40)
    """
    # Adaylar = Yüksek + Orta
    cand = df_agg[df_agg["tier"].isin(["Yüksek", "Orta"])].copy()
    if cand.empty:
        return {"zones": []}

    merged  = cand.merge(GEO_DF, on=KEY_COL)
    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy()

    k = max(1, min(int(k_planned), 50))
    cents, assign = kmeans_like(coords, weights, k)

    # Hücre kapasitesi (adet) ~ görev süresi / (hücre_süresi * (1+overhead))
    cap_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + travel_overhead))))

    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy().sort_values("expected", ascending=False)

        # kapasite kadar en riskli hücreyi al
        sub_planned = sub.head(cap_cells).copy()

        cz = cents[z]
        # rotayı açıya göre sırala
        angles = np.arctan2(sub_planned["centroid_lat"] - cz[1], sub_planned["centroid_lon"] - cz[0])
        sub_planned = sub_planned.assign(angle=angles).sort_values("angle")

        route = sub_planned[["centroid_lat", "centroid_lon"]].to_numpy().tolist()
        n_cells = len(sub_planned)
        eta_minutes = int(round(n_cells * cell_minutes * (1.0 + travel_overhead)))
        util = min(100, int(round(100 * eta_minutes / max(1, duty_minutes))))

        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz[1]), "lon": float(cz[0])},
            "cells": sub_planned[KEY_COL].astype(str).tolist(),
            "route": route,
            "expected_risk": float(sub_planned["expected"].mean()),
            "planned_cells": int(n_cells),
            "eta_minutes": int(eta_minutes),
            "utilization_pct": int(util),
            "capacity_cells": int(cap_cells),
        })
    return {"zones": zones}
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def color_for_tier(tier: str) -> str:
    return {"Yüksek": "#d62728", "Orta": "#ff7f0e", "Hafif": "#1f77b4"}.get(tier, "#1f77b4")

def build_map(
    df_agg: pd.DataFrame,
    patrol: Dict | None = None,
    show_popups: bool = True
) -> folium.Map:
    # Boş/eksik durumda güvenli çık
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty or KEY_COL not in df_agg.columns:
        return m

    # Boyama ve "acil" eşiği expected üzerinden
    values = df_agg["expected"].to_numpy()

    # Hızlı erişim için index
    idx = df_agg.set_index(KEY_COL)

    for feat in GEO_FEATURES:
        gid = feat["properties"].get("id")  # = geoid
        if gid not in idx.index:
            continue

        r = idx.loc[gid]
        expected = float(r["expected"])
        tier     = str(r["tier"])
        q10      = float(r["q10"])
        q90      = float(r["q90"])
        types    = [(t, float(r[t])) for t in CRIME_TYPES]

        # en olası 3 suç
        top3 = sorted(types, key=lambda x: x[1], reverse=True)[:3]
        top_html = "".join(f"<li>{t}: {v:.2f}</li>" for t, v in top3)

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

        # GeoJSON'u ekle, popup'ı (opsiyonel) sonradan bağla
        geo = folium.GeoJson(
            data=feat,
            style_function=lambda _x, s=style: s,
            highlight_function=lambda _x: {"weight": 1.5, "color": "#000000", "fillOpacity": 0.7},
            tooltip=folium.Tooltip(f"{gid} — E[olay]: {expected:.2f} — {tier}"),
        )
        # Popup: sadece Yüksek/Orta için ekle (istersen 'or True' yapıp hepsine aç)
        if show_popups: 
            folium.Popup(popup_html, max_width=280).add_to(geo)
        geo.add_to(m)

    # En yüksek %1 beklenen olaya kırmızı uyarı
    if len(values):
        thr99 = np.quantile(values, 0.99)
        urgent = df_agg[df_agg["expected"] >= thr99]
        merged = urgent.merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL, how="left")
        for _, rr in merged.iterrows():
            if pd.isna(rr.get("centroid_lat")) or pd.isna(rr.get("centroid_lon")):
                continue
            folium.CircleMarker(
                location=[float(rr["centroid_lat"]), float(rr["centroid_lon"])],
                radius=6,
                color="#000",
                fill=True,
                fill_color="#ff0000",
                popup=folium.Popup("ACİL — üst %1 E[olay]", max_width=150),
                interactive=False,
            ).add_to(m)

    # Devriye rotaları
    if patrol and patrol.get("zones"):
        for z in patrol["zones"]:
            folium.PolyLine(z["route"], tooltip=f"{z['id']} rota").add_to(m)
            folium.Marker(
                [z["centroid"]["lat"], z["centroid"]["lon"]],
                icon=folium.DivIcon(
                    html=(
                        "<div style='background:#111;color:#fff;padding:2px 6px;"
                        "border-radius:6px'> {}</div>".format(z["id"])
                    )
                ),
            ).add_to(m)

    return m
    
def build_map_fast(df_agg: pd.DataFrame, show_popups: bool = False, patrol: Dict | None = None) -> folium.Map:
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles="cartodbpositron")
    if df_agg is None or df_agg.empty:
        return m

    # Görselleştirme renkleri
    color_map = {r[KEY_COL]: color_for_tier(r["tier"]) for _, r in df_agg.iterrows()}

    # Veri sözlüğü (popup için hızlı erişim)
    data_map = df_agg.set_index(KEY_COL).to_dict(orient="index")

    # GEOJSON’u popup içeriğiyle zenginleştir
    features = []
    for feat in GEO_FEATURES:
        # derin kopya (global objeyi kirletmeyelim)
        f = json.loads(json.dumps(feat))
        gid = f["properties"].get("id")

        row = data_map.get(gid)
        if row:
            expected = float(row["expected"])
            tier     = str(row["tier"])
            q10      = float(row["q10"])
            q90      = float(row["q90"])
            types    = {t: float(row[t]) for t in CRIME_TYPES}
            top3     = sorted(types.items(), key=lambda x: x[1], reverse=True)[:3]
            top_html = "".join([f"<li>{t}: {v:.2f}</li>" for t, v in top3])

            popup_html = (
                f"<b>{gid}</b><br/>"
                f"E[olay] (ufuk): {expected:.2f} &nbsp;•&nbsp; Öncelik: <b>{tier}</b><br/>"
                f"<b>En olası 3 tip</b>"
                f"<ul style='margin-left:12px'>{top_html}</ul>"
                f"<i>Belirsizlik (saatlik ort.): q10={q10:.2f}, q90={q90:.2f}</i>"
            )
            f["properties"]["popup_html"] = popup_html
            f["properties"]["expected"]   = round(expected, 2)
            f["properties"]["tier"]       = tier
        features.append(f)

    fc = {"type": "FeatureCollection", "features": features}

    def style_fn(feat):
        gid = feat["properties"].get("id")
        return {
            "fillColor": color_map.get(gid, "#9ecae1"),
            "color": "#666666",
            "weight": 0.3,
            "fillOpacity": 0.55,
        }

    # Tooltip + Popup (opsiyonel)
    tooltip = folium.GeoJsonTooltip(
        fields=["id", "tier", "expected"],
        aliases=["GEOID", "Öncelik", "E[olay]"],
        localize=True,
        sticky=False,
    ) if show_popups else None

    popup = folium.GeoJsonPopup(
        fields=["popup_html"],
        labels=False,
        parse_html=False,
        max_width=280,
    ) if show_popups else None

    gj = folium.GeoJson(fc, style_function=style_fn, tooltip=tooltip, popup=popup)
    gj.add_to(m)

    # En yüksek %1 beklenen olaya kırmızı uyarı
    thr99 = np.quantile(df_agg["expected"].to_numpy(), 0.99)
    urgent = df_agg[df_agg["expected"] >= thr99]
    merged = urgent.merge(GEO_DF[[KEY_COL, "centroid_lat", "centroid_lon"]], on=KEY_COL)
    for _, r in merged.iterrows():
        folium.CircleMarker(
            location=[r["centroid_lat"], r["centroid_lon"]],
            radius=5, color="#000", fill=True, fill_color="#ff0000",
            popup=None if not show_popups else folium.Popup("ACİL — üst %1 E[olay]", max_width=150)
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

# =============================
# UI — SİDEBAR
# =============================
st.sidebar.markdown("### Görünüm")
sekme = st.sidebar.radio("", options=["Operasyon", "Raporlar"], index=0, horizontal=True)
st.sidebar.divider()

st.sidebar.header("Ayarlar")
ufuk = st.sidebar.radio("Ufuk", options=["24s", "48s", "7g"], index=0, horizontal=True)

# Aralık seçimi (başlangıç-bitiş)
if ufuk == "24s":
    max_h, step = 24, 1
elif ufuk == "48s":
    max_h, step = 48, 3
else:
    max_h, step = 7 * 24, 24

start_h, end_h = st.sidebar.slider(
    "Zaman aralığı (şimdiden + saat)",
    min_value=0, max_value=max_h, value=(0, max_h), step=step
)

st.sidebar.divider()
st.sidebar.subheader("Devriye Parametreleri")
K_planned = st.sidebar.number_input("Planlanan devriye sayısı (K)", min_value=1, max_value=50, value=6, step=1)
duty_minutes = st.sidebar.number_input("Devriye görev süresi (dk)", min_value=15, max_value=600, value=120, step=15)
cell_minutes = st.sidebar.number_input("Hücre başına ort. kontrol (dk)", min_value=2, max_value=30, value=6, step=1)

# Devriye butonu (K/band yok)
colA, colB = st.sidebar.columns(2)
btn_predict = colA.button("Tahmin et")
btn_patrol  = colB.button("Devriye öner", disabled=st.session_state.get("agg") is None)
show_popups = st.sidebar.checkbox(
    "Hücre popup'larını (en olası 3 suç) göster",
    value=True,
    help="Harita hücresine tıklayınca beklenen dağılıma göre ilk 3 suç tipini gösterir."
)

st.sidebar.caption("• Tahmin et: seçtiğin aralık için riskleri hesaplar.  • Devriye öner: K ekip ve görev süresine göre kümeler/rota üretir.")

# =============================
# STATE
# =============================
if "forecast" not in st.session_state:
    st.session_state["forecast"] = None
    st.session_state["agg"] = None
    st.session_state["patrol"] = None

# =============================
# ANA BÖLÜM
# =============================
if sekme == "Operasyon":
    col1, col2 = st.columns([2.4, 1.0])
    
    with col1:
        st.caption(f"Son güncelleme (SF): {now_sf_iso()}")
    
        if btn_predict or st.session_state.get("agg") is None:
            start_dt = (
                datetime.utcnow()
                + timedelta(hours=SF_TZ_OFFSET + start_h)
            ).replace(minute=0, second=0, microsecond=0)
    
            horizon_h = max(1, end_h - start_h)
            start_iso = start_dt.isoformat()
    
            # Hızlı agregasyon
            agg = aggregate_fast(start_iso, horizon_h)
    
            st.session_state["forecast"] = None
            st.session_state["agg"] = agg
            st.session_state["patrol"] = None
            st.session_state["start_iso"] = start_iso
            st.session_state["horizon_h"] = horizon_h
    
        agg = st.session_state.get("agg")
        
        if agg is not None:
            # haritayı ÇİZ ve DÖNÜŞ değerini AL (tek çağrı!)
            m = build_map_fast(
                agg,
                show_popups=show_popups,
                patrol=st.session_state.get("patrol")
            )
            ret = st_folium(
                m,
                key="riskmap",   # Stabil anahtar EKLE
                width=None,
                height=540,
                returned_objects=["last_object_clicked", "last_clicked"]
            )
            
            clicked_gid = None
            if ret:
                clicked_gid, _ = resolve_clicked_gid(ret)
                
            clicked_gid = None
            if ret:
                # 1) GeoJSON/çizim objesinden dene
                obj = ret.get("last_object_clicked") or ret.get("last_active_drawing")
                if isinstance(obj, dict):
                    # Bazı sürümlerde id kök seviyede gelir
                    clicked_gid = str(obj.get("id") or "") or None
                    if not clicked_gid:
                        props = (obj.get("properties")
                                 or obj.get("feature", {}).get("properties", {})
                                 or {})
                        clicked_gid = props.get("id") or props.get(KEY_COL) or props.get("GEOID")
            
                # 2) Olmazsa yalnız koordinattan en yakın hücre
                if not clicked_gid:
                    latlon = _extract_latlon_from_ret(ret)
                    if latlon:
                        lat, lon = latlon
                        clicked_gid = nearest_geoid(lat, lon)

            # açıklama için gerekli zaman bilgisi
            start_iso  = st.session_state.get("start_iso")
            horizon_h  = st.session_state.get("horizon_h")
            if (start_iso is None) or (horizon_h is None):
                # butona basılmadan önce de çalışabilsin diye emniyet
                start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET + start_h)).replace(minute=0, second=0, microsecond=0)
                start_iso = start_dt.isoformat()
                horizon_h = max(1, end_h - start_h)
        
            # tıklanınca state'e açıklamayı yaz
            if clicked_gid:
                st.session_state["explain"] = {
                    "geoid": clicked_gid,
                    "data": local_explain(agg, clicked_gid, start_iso, horizon_h),
                }

            with st.expander("Hızlı inceleme"):
                top_ids = top_risky_table(agg, n=20)[KEY_COL].astype(str).tolist()
                pick = st.selectbox("Bölge seç (alternatif):", ["—"] + top_ids, index=0, key="pick_alt")
                if pick != "—":
                    st.session_state["explain"] = {
                        "geoid": pick,
                        "data": local_explain(agg, pick, start_iso, horizon_h),
                    }
        
            # --- SOLA (haritanın ALTINA) AÇIKLAMA PANELİ ---
            info = st.session_state.get("explain")
            start_iso = st.session_state.get("start_iso")
            horizon_h = st.session_state.get("horizon_h")
            
            if info and info.get("geoid") and start_iso and horizon_h:
                render_result_card(st.session_state["agg"], info["geoid"], start_iso, horizon_h)
            else:
                st.info("Haritada bir hücreye tıklayın veya listeden seçin; kart burada görünecek.")
        else:
            st.info("Önce ‘Tahmin et’ ile bir tahmin üretin.")
    
    with col2:
        st.subheader("Risk Özeti")
        if st.session_state["agg"] is not None:
            a = st.session_state["agg"]
            kpi_expected = round(float(a["expected"].sum()), 2)
            high = int((a["tier"] == "Yüksek").sum())
            mid  = int((a["tier"] == "Orta").sum())
            low  = int((a["tier"] == "Hafif").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beklenen olay (ufuk)", kpi_expected)
            c2.metric("Yüksek", high); c3.metric("Orta", mid); c4.metric("Düşük", low)
    
            with st.expander("Öncelik kümeleri — geoid listeleri"):
                cc1, cc2, cc3 = st.columns(3)
                cc1.write(", ".join(a.loc[a["tier"]=="Yüksek", KEY_COL].astype(str).tolist()) or "—")
                cc2.write(", ".join(a.loc[a["tier"]=="Orta",   KEY_COL].astype(str).tolist()) or "—")
                cc3.write(", ".join(a.loc[a["tier"]=="Hafif",  KEY_COL].astype(str).tolist()) or "—")
    
        st.subheader("En riskli bölgeler")
        if st.session_state["agg"] is not None:
            st.dataframe(top_risky_table(st.session_state["agg"]), use_container_width=True, height=300)
    
        st.subheader("Devriye özeti")
        if st.session_state.get("agg") is not None and btn_patrol:
            st.session_state["patrol"] = allocate_patrols(
                st.session_state["agg"],
                k_planned=K_planned,
                duty_minutes=int(duty_minutes),
                cell_minutes=int(cell_minutes),
                travel_overhead=0.40,
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
    
        st.subheader("Dışa aktar")
        if st.session_state["agg"] is not None:
            csv = st.session_state["agg"].to_csv(index=False).encode("utf-8")
            st.download_button("CSV indir", data=csv,
                               file_name=f"risk_export_{int(time.time())}.csv",
                               mime="text/csv")

elif sekme == "Raporlar":
    st.header("Raporlar")

    # Operasyon sekmesindeki aralığı kullan; yoksa varsayılan
    start_iso = st.session_state.get("start_iso")
    horizon_h = st.session_state.get("horizon_h")
    if (start_iso is None) or (horizon_h is None):
        start_dt = (datetime.utcnow() + timedelta(hours=SF_TZ_OFFSET)).replace(minute=0, second=0, microsecond=0)
        start_iso = start_dt.isoformat(); horizon_h = 24

    # Tahmin tablosu yoksa üret
    agg = st.session_state.get("agg")
    if agg is None:
        agg = aggregate_fast(start_iso, horizon_h)
        st.session_state["agg"] = agg

    tab1, tab2, tab3 = st.tabs(["Vardiya Brifi", "Haftalık Özet", "Model Sağlık Kartı"])

    # -------- VARDİYA BRİFİ --------
    with tab1:
        st.caption(f"Aralık: {start_iso} +{horizon_h} saat")
        top5 = top_risky_table(agg, n=5)

        lines = []
        for gid in top5[KEY_COL].astype(str).tolist():
            ex = local_explain(agg, gid, start_iso, horizon_h)
            lead1 = ex["top_types"][0][0] if ex["top_types"] else "-"
            lead2 = ex["top_types"][1][0] if len(ex["top_types"])>1 else "-"
            tips  = actionable_cues(ex["top_types"])
            conf  = confidence_label(ex["q10"], ex["q90"])
            win   = risk_window_text(start_iso, horizon_h)
            lines.append(
                f"{gid}: {win} aralığında risk **{ex['tier'].lower()}**. "
                f"Öncelik: {lead1}, {lead2}. E[olay]={ex['expected']:.1f} (güven {conf}). "
                f"Ekip odak: {tips[0] if tips else '-'}."
            )

        patrol = st.session_state.get("patrol") or allocate_patrols(
            agg, k_planned=6, duty_minutes=120, cell_minutes=6, travel_overhead=0.40
        )
        patrol_rows = [{
            "zone": z["id"],
            "cells_planned": z["planned_cells"],
            "capacity_cells": z["capacity_cells"],
            "eta_minutes": z["eta_minutes"],
            "avg_risk(E[olay])": round(z["expected_risk"], 2),
        } for z in patrol.get("zones", [])]

        st.subheader("Top-5 bölge")
        st.dataframe(top5, use_container_width=True, height=200)
        st.subheader("Radyo anonsları")
        for ln in lines:
            st.code(ln, language=None)
        st.subheader("Önerilen devriye")
        if patrol_rows:
            st.dataframe(pd.DataFrame(patrol_rows), use_container_width=True, height=200)
        else:
            st.info("Devriye önerisi yok (aday hücre bulunamadı).")

        html = brief_html(top5, lines, patrol_rows, start_iso, horizon_h).encode("utf-8")
        st.download_button("Vardiya Brifi – HTML indir", data=html,
                           file_name="vardiya_brifi.html", mime="text/html")
        st.caption("İpucu: Açıp Ctrl/Cmd+P → 'PDF olarak kaydet'.")

    # -------- HAFTALIK OPERASYON ÖZETİ --------
    with tab2:
        events = load_events("data/events.csv")
        if events.empty:
            st.warning("`data/events.csv` yok/boş: örnek metrikler gerçek olay olmadan hesaplanamaz.")
            top10 = top_risky_table(agg, n=10)
            html = weekly_html(float("nan"), 0.0, top10).encode("utf-8")
            st.dataframe(top10, use_container_width=True, height=260)
            st.download_button("Haftalık Özet – HTML indir", data=html,
                               file_name="haftalik_ozet.html", mime="text/html")
        else:
            ev_win = slice_events(events, start_iso, horizon_h)
            eval_df = prep_eval_frames(agg, ev_win)
            pai10, share10 = pai_at_k(eval_df, 10)
            st.metric("PAI@10", "—" if np.isnan(pai10) else f"{pai10:.2f}")
            top10 = top_risky_table(agg, n=10)
            st.dataframe(top10, use_container_width=True, height=260)

            d = eval_df.sort_values("p", ascending=False).reset_index(drop=True)
            d["alan_pay"] = (np.arange(len(d)) + 1) / len(d)
            total_events = max(1, d["y_count"].sum())
            d["olay_kum"] = d["y_count"].cumsum()
            d["olay_pay"] = d["olay_kum"] / total_events
            cap_chart = alt.Chart(d).mark_line(point=True).encode(
                x=alt.X("alan_pay:Q", title="Alan payı", axis=alt.Axis(format="%")),
                y=alt.Y("olay_pay:Q", title="Olay payı", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("alan_pay:Q", format=".0%"), alt.Tooltip("olay_pay:Q", format=".0%")]
            ).properties(height=220)
            st.altair_chart(cap_chart, use_container_width=True)

            html = weekly_html(pai10, share10, top10).encode("utf-8")
            st.download_button("Haftalık Özet – HTML indir", data=html,
                               file_name="haftalik_ozet.html", mime="text/html")

    # -------- MODEL SAĞLIK KARTI --------
    with tab3:
        events = load_events("data/events.csv")
        if events.empty:
            st.info("Gerçek olay olmadan Brier/kapsama hesaplanamaz.")
        else:
            ev_win = slice_events(events, start_iso, horizon_h)
            eval_df = prep_eval_frames(agg, ev_win)
            brier = float(np.mean((eval_df["p"] - eval_df["y"]) ** 2))
            low = np.maximum(0.0, eval_df["p"] - 0.08)
            high = np.minimum(1.0, eval_df["p"] + 0.08)
            coverage = float(((eval_df["y"] >= low) & (eval_df["y"] <= high)).mean())
            st.metric("Brier", f"{brier:.3f}")
            st.metric("Kapsama (q10–q90±0.08)", f"{coverage:.0%}")

            bins = np.linspace(0, 1, 11)
            d = eval_df.copy()
            d["bin"] = np.digitize(d["p"], bins, right=True) - 1
            reli = d.groupby("bin", as_index=False).agg(
                p_hat=("p","mean"), y_rate=("y","mean"), n=("y","size")
            ).dropna()
            chart = alt.Chart(reli).mark_line(point=True).encode(
                x=alt.X("p_hat:Q", title="Tahmin olasılığı"),
                y=alt.Y("y_rate:Q", title="Gerçekleşme oranı"),
                tooltip=["p_hat","y_rate","n"]
            ).properties(height=220)
            diag = alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_rule().encode(x="x", y="y")
            st.altair_chart(chart + diag, use_container_width=True)
