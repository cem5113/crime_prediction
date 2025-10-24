# components/utils/hotspots.py
from __future__ import annotations

import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Paket içi sabitler (göreli import)
try:
    from .constants import KEY_COL as DEFAULT_KEY_COL
except Exception:
    DEFAULT_KEY_COL = "geoid"

try:
    from .constants import DAYS  # ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
except Exception:
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# sklearn BallTree opsiyonel (Cloud'da olmayabilir)
try:
    from sklearn.neighbors import BallTree  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ────────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ────────────────────────────────────────────────────────────────────────────────
def _to_utc_datetime(s: pd.Series) -> pd.Series:
    """Kolonları güvenli UTC datetime'a çevirir."""
    return pd.to_datetime(s, utc=True, errors="coerce")


def _ensure_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """lat/lon alias isimlerini normalize eder."""
    out = df.copy()
    lower = {str(c).lower(): c for c in out.columns}

    if lat_col not in out.columns:
        if "lat" in lower:
            out = out.rename(columns={lower["lat"]: lat_col})
        elif "latitude" in lower:
            out = out.rename(columns={lower["latitude"]: lat_col})

    if lon_col not in out.columns:
        if "lon" in lower:
            out = out.rename(columns={lower["lon"]: lon_col})
        elif "longitude" in lower:
            out = out.rename(columns={lower["longitude"]: lon_col})

    return out


def _haversine_dist_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vektörize haversine: (deg) → metre."""
    R = 6_371_000.0  # m
    φ1 = np.radians(lat1)
    φ2 = np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ────────────────────────────────────────────────────────────────────────────────
# Geçici Hotspot: Gaussian (uzaysal) + Half-life (zamansal) ağırlık
# ────────────────────────────────────────────────────────────────────────────────
def temp_hotspot_scores(
    events: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    lookback_h: int = 48,
    sigma_m: int = 500,
    half_life_h: int = 24,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ts_col: str = "timestamp",
    centroid_lat_col: str = "centroid_lat",
    centroid_lon_col: str = "centroid_lon",
    key_col: Optional[str] = None,
    type_col: str = "type",                     # olay türü (assault, theft, …)
    category: Optional[str] = None,             # None/"all" => hepsi; aksi tek kategori
    hours_filter: Optional[Tuple[int, int]] = None,  # (h1,h2) [h1, h2)
) -> pd.DataFrame:
    """
    Son 'lookback_h' saat içindeki olaylara bakarak her hücre için geçici hotspot puanı hesaplar.
    Uzayda Gaussian(σ=sigma_m), zamanda half-life (half_life_h) sönümü kullanılır.

    Dönüş: [key_col, hotspot_raw, hotspot_score(0-1)]
    """
    kcol = key_col or DEFAULT_KEY_COL

    # Çıkış iskeleti
    out = geo_df[[kcol]].copy()
    out["hotspot_raw"] = 0.0
    out["hotspot_score"] = 0.0

    if events is None or events.empty or geo_df.empty:
        return out

    # Koordinat alias düzeltmeleri
    ev = _ensure_latlon(events.copy(), lat_col, lon_col)

    # Kategori filtresi
    if category and str(category).lower() not in ("all", "tüm", "tum"):
        if type_col in ev.columns:
            ev = ev[ev[type_col].astype(str).str.lower() == str(category).lower()]

    # Zaman filtresi
    now = pd.Timestamp.utcnow()
    ev[ts_col] = _to_utc_datetime(ev[ts_col]) if ts_col in ev.columns else pd.NaT
    ev = ev.dropna(subset=[lat_col, lon_col, ts_col])
    if ev.empty:
        return out

    ev = ev[(now - ev[ts_col]) <= pd.Timedelta(hours=lookback_h)]
    if hours_filter:
        h1, h2 = hours_filter
        h2 = (h2 - 1) % 24
        ev = ev[ev[ts_col].dt.hour.between(h1, h2)]
    if ev.empty:
        return out

    # Zamansal ağırlık (half-life)
    dt_h = (now - ev[ts_col]).dt.total_seconds() / 3600.0
    w_t = np.power(2.0, -dt_h / float(max(half_life_h, 1e-6)))

    # GEOID merkezleri
    if centroid_lat_col not in geo_df.columns or centroid_lon_col not in geo_df.columns:
        # Merkez yoksa skor veremeyiz
        return out
    geo_lat = geo_df[centroid_lat_col].to_numpy(dtype=float)
    geo_lon = geo_df[centroid_lon_col].to_numpy(dtype=float)

    ev_lat = ev[lat_col].to_numpy(dtype=float)
    ev_lon = ev[lon_col].to_numpy(dtype=float)

    hotspot_raw = np.zeros(len(geo_df), dtype=float)

    if HAS_SKLEARN:
        # BallTree (haversine)
        R = 6_371_000.0
        ev_rad = np.radians(np.c_[ev_lat, ev_lon])
        geo_rad = np.radians(np.c_[geo_lat, geo_lon])
        tree = BallTree(ev_rad, metric="haversine")
        query_r = 3 * (sigma_m / R)  # 3σ (radyan)

        ind = tree.query_radius(geo_rad, r=query_r, return_distance=True)
        for gi, (idxs, dists_rad) in enumerate(zip(*ind)):
            if len(idxs) == 0:
                continue
            d_m = dists_rad * R
            w_s = np.exp(-0.5 * (d_m / float(sigma_m)) ** 2)
            hotspot_raw[gi] = float((w_s * w_t.iloc[idxs].to_numpy()).sum())
    else:
        # Numpy fallback: her geoid için çevredeki olayları tara (yavaş ama bağımlılıksız)
        # Önce kaba bir sınır kutusu (yaklaşık 3σ ≈ 3*sigma_m)
        approx_deg = (3 * float(sigma_m)) / 111_000.0  # ~metre→derece
        for gi, (clat, clon) in enumerate(zip(geo_lat, geo_lon)):
            # Hızlı ön-seçim
            mask = (
                (ev_lat >= clat - approx_deg) & (ev_lat <= clat + approx_deg) &
                (ev_lon >= clon - approx_deg) & (ev_lon <= clon + approx_deg)
            )
            if not np.any(mask):
                continue
            lat_sub = ev_lat[mask]
            lon_sub = ev_lon[mask]
            w_t_sub = w_t.to_numpy()[mask]
            d_m = _haversine_dist_m(clat, clon, lat_sub, lon_sub)
            # 3σ dışında kalanları zaten ön-seçimle kırptık, yine de güvenlik:
            use = d_m <= (3 * float(sigma_m))
            if not np.any(use):
                continue
            d_sel = d_m[use]
            w_sel = w_t_sub[use]
            w_s = np.exp(-0.5 * (d_sel / float(sigma_m)) ** 2)
            hotspot_raw[gi] = float(np.sum(w_s * w_sel))

    # 0-1 skor (robust ölçek)
    if hotspot_raw.max() > 0:
        p99 = np.percentile(hotspot_raw, 99)
        scale = p99 if p99 > 0 else float(hotspot_raw.max())
    else:
        scale = 1.0

    out["hotspot_raw"] = hotspot_raw
    out["hotspot_score"] = np.clip(hotspot_raw / (scale + 1e-12), 0, 1)
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Folium/pydeck için geçici hotspot nokta seti (opsiyonel yardımcı)
# ────────────────────────────────────────────────────────────────────────────────
def make_temp_hotspot_points(
    events: pd.DataFrame,
    *,
    lookback_h: int = 48,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    ts_col: str = "timestamp",
    type_col: str = "type",
    category: Optional[str] = None,
    hours_filter: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Folium HeatMap/pydeck için (lat,lon,weight) nokta DataFrame'i üretir.
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=[lat_col, lon_col, "weight"])

    ev = _ensure_latlon(events.copy(), lat_col, lon_col)
    now = pd.Timestamp.utcnow()

    # Zaman/kategori filtreleri
    if ts_col in ev.columns:
        ev[ts_col] = _to_utc_datetime(ev[ts_col])
        ev = ev[(now - ev[ts_col]) <= pd.Timedelta(hours=lookback_h)]
        if hours_filter:
            h1, h2 = hours_filter
            h2 = (h2 - 1) % 24
            ev = ev[ev[ts_col].dt.hour.between(h1, h2)]
    if category and type_col in ev.columns:
        ev = ev[ev[type_col].astype(str).str.lower() == str(category).lower()]

    ev = ev.dropna(subset=[lat_col, lon_col])
    if ev.empty:
        return pd.DataFrame(columns=[lat_col, lon_col, "weight"])

    out = ev[[lat_col, lon_col]].copy()
    out["weight"] = 1.0
    return out


# ────────────────────────────────────────────────────────────────────────────────
# Gün × Saat ısı matrisi (Streamlit + Matplotlib)
# ────────────────────────────────────────────────────────────────────────────────
def _ensure_day_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    'hour' ve 'dow' kolonlarını güvenli şekilde elde etmeye çalışır.
    'expected' varsa 'value' olarak kullanır, yoksa score/weight/1.0.
    """
    out = df.copy()

    # hour/dow türetmeleri
    if "hour" not in out.columns and "ts" in out.columns:
        ts = _to_utc_datetime(out["ts"])
        out["hour"] = ts.dt.hour
        out["dow"] = ts.dt.day_name().str[:3]  # Mon, Tue, ...
    if "hour" not in out.columns and "start_hour" in out.columns:
        out["hour"] = out["start_hour"].astype(int).clip(0, 23)
    if "dow" not in out.columns:
        out["dow"] = "Mon"

    # değer kolonu
    if "value" not in out.columns:
        if "expected" in out.columns:
            out["value"] = out["expected"].astype(float)
        elif "score" in out.columns:
            out["value"] = out["score"].astype(float)
        elif "weight" in out.columns:
            out["value"] = out["weight"].astype(float)
        else:
            out["value"] = 1.0

    return out


def render_day_hour_heatmap(
    df_agg: pd.DataFrame,
    start_iso: Optional[str] = None,
    horizon_h: Optional[int] = None,
) -> None:
    """
    app.py'de çağrılan ısı matrisi renderer.
    df_agg: aggregate/pred çıktısı (satırlarda geoid bazlı skorlar veya olaylar)
    start_iso/horizon_h sadece başlık bilgisinde kullanılır.
    """
    if not isinstance(df_agg, pd.DataFrame) or df_agg.empty:
        st.caption("Isı matrisi için veri yok.")
        return

    df = _ensure_day_hour(df_agg)

    if "hour" not in df.columns or "dow" not in df.columns:
        st.info("Isı matrisi için 'hour' ve 'dow' alanları bulunamadı.")
        return

    sub = df[["dow", "hour", "value"]].copy()
    sub["dow"] = pd.Categorical(sub["dow"], categories=DAYS, ordered=True)

    piv = sub.groupby(["dow", "hour"], as_index=False)["value"].sum()
    tbl = piv.pivot(index="dow", columns="hour", values="value").reindex(index=DAYS)
    tbl = tbl.fillna(0.0)

    # Grafik (seaborn YOK, tek plot, renk ayarı yapılmıyor)
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=120)
    im = ax.imshow(tbl.values, aspect="auto")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(x) for x in range(0, 24, 2)])
    ax.set_yticks(range(len(tbl.index)))
    ax.set_yticklabels(tbl.index)
    ax.set_xlabel("Saat")
    ax.set_ylabel("Gün")

    title = "Gün × Saat ısı matrisi"
    if start_iso and horizon_h:
        title += f" • Başlangıç: {start_iso} • Ufuk: {horizon_h}h"
    ax.set_title(title, fontsize=11)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Yoğunluk", rotation=90, labelpad=10)

    st.pyplot(fig, clear_figure=True)
