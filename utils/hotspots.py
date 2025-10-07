# utils/hotspots.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

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
    key_col: str = "GEOID",
    type_col: str = "type",                 # olay türü sütunu (örn. "assault", "theft", ...)
    category: str | None = None,            # None/"all" => hepsi, yoksa tek kategoriye filtre
    hours_filter: tuple[int, int] | None = None  # (start,end) [h1,h2) ör. (9,18)
) -> pd.DataFrame:
    """
    Son 'lookback_h' saatlik olaylardan, GEOID merkezlerine uzaysal (σ=sigma_m, Gaussian)
    ve zamansal (half-life) ağırlıklarla geçici hotspot puanı üretir.
    Döndürür: [key_col, hotspot_raw, hotspot_score (0-1)]
    """
    out = geo_df[[key_col]].copy()
    out["hotspot_raw"] = 0.0
    out["hotspot_score"] = 0.0

    if events is None or events.empty:
        return out

    ev = events.copy()
    # Kategori filtresi
    if category and str(category).lower() not in ("all", "tüm", "tum"):
        if type_col in ev.columns:
            ev = ev[ev[type_col] == category]

    # Zaman filtresi
    now = pd.Timestamp.utcnow()
    ev[ts_col] = pd.to_datetime(ev[ts_col], utc=True, errors="coerce")
    ev = ev[(now - ev[ts_col]) <= pd.Timedelta(hours=lookback_h)]
    if hours_filter:
        h1, h2 = hours_filter
        h2 = (h2 - 1) % 24
        ev = ev[ev[ts_col].dt.hour.between(h1, h2)]

    ev = ev.dropna(subset=[lat_col, lon_col, ts_col])
    if ev.empty:
        return out

    # BallTree (haversine)
    R = 6_371_000.0
    ev_rad = np.radians(ev[[lat_col, lon_col]].to_numpy())
    geo_rad = np.radians(geo_df[["centroid_lat", "centroid_lon"]].to_numpy())
    tree = BallTree(ev_rad, metric="haversine")
    query_r = 3 * (sigma_m / R)  # 3σ yarıçap (radyan)

    # Zamansal ağırlık
    dt_h = (now - ev[ts_col]).dt.total_seconds() / 3600.0
    w_t = np.power(2.0, -dt_h / float(half_life_h))  # half-life sönümü

    hotspot_raw = np.zeros(len(geo_df), dtype=float)
    ind = tree.query_radius(geo_rad, r=query_r, return_distance=True)

    for gi, (idxs, dists_rad) in enumerate(zip(*ind)):
        if len(idxs) == 0:
            continue
        d_m = dists_rad * R
        w_s = np.exp(-0.5 * (d_m / float(sigma_m)) ** 2)
        hotspot_raw[gi] = float((w_s * w_t.iloc[idxs].to_numpy()).sum())

    p99 = np.percentile(hotspot_raw, 99) if hotspot_raw.max() > 0 else 1.0
    out["hotspot_raw"] = hotspot_raw
    out["hotspot_score"] = np.clip(hotspot_raw / (p99 + 1e-12), 0, 1)
    return out
