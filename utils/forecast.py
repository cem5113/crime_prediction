# utils/forecast.py
from __future__ import annotations
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.constants import CRIME_TYPES, KEY_COL

# ---- küçük yardımcılar ----
def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

# -------------------- Baz yoğunluk: normalize --------------------
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()
    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    raw = 0.2 + 0.8 * (peak1 + peak2) + 0.07
    raw = raw - raw.min()
    return raw / (raw.max() + 1e-9)  # 0..1

# -------------------- Near-Repeat (NR) skor vektörü --------------------
def _near_repeat_score(
    geo_df: pd.DataFrame,
    events: pd.DataFrame,
    start_iso: str,
    *,
    lookback_h: int = 24,
    spatial_radius_m: int = 400,
    temporal_decay_h: float = 12.0,
) -> np.ndarray:
    """
    Her hücre için 0..1 arası NR skoru üretir.
    Yakın (mesafe & zaman) olaylar daha yüksek ağırlık alır.
    """
    if events is None or events.empty:
        return np.zeros(len(geo_df), dtype=float)

    start = datetime.fromisoformat(start_iso)
    t0 = start - timedelta(hours=lookback_h)
    ev = events.loc[(events["ts"] >= t0) & (events["ts"] <= start)].copy()
    if ev.empty or not {"lat", "lon"}.issubset(ev.columns):
        return np.zeros(len(geo_df), dtype=float)

    cent_lat = geo_df["centroid_lat"].to_numpy()
    cent_lon = geo_df["centroid_lon"].to_numpy()

    # vektörel hesap: her olay için tüm centroid mesafeleri
    nr = np.zeros(len(geo_df), dtype=float)
    tau = max(temporal_decay_h, 1e-6)

    for _, r in ev.iterrows():
        d = _haversine_m(cent_lat, cent_lon, float(r["lat"]), float(r["lon"]))
        w_space = np.exp(-np.maximum(d, 0.0) / spatial_radius_m)
        dt_h = (start - r["ts"]).total_seconds() / 3600.0
        w_time = np.exp(-max(dt_h, 0.0) / tau)
        nr += w_space * w_time

    # 0..1’a sıkıştır
    nr = nr - nr.min()
    maxv = nr.max()
    return (nr / maxv) if maxv > 1e-9 else np.zeros_like(nr)

# -------------------- Hızlı agregasyon (NR dahil) --------------------
def aggregate_fast(
    start_iso: str,
    horizon_h: int,
    geo_df: pd.DataFrame,
    base_int: np.ndarray,
    *,
    k_lambda: float = 0.12,       # saatlik ölçek
    events: pd.DataFrame | None = None,
    near_repeat_alpha: float = 0.35,  # NR katkı gücü (0..1 tipik)
    nr_lookback_h: int = 24,
    nr_radius_m: int = 400,
    nr_decay_h: float = 12.0,
) -> pd.DataFrame:
    """
    Saatlik λ = k_lambda * base_int * diurnal * (1 + near_repeat_alpha * NR)
    expected = saatlik λ’ların toplamı
    """
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    # Near-Repeat skoru (0..1)
    nr = _near_repeat_score(
        geo_df, events, start_iso,
        lookback_h=nr_lookback_h,
        spatial_radius_m=nr_radius_m,
        temporal_decay_h=nr_decay_h,
    )

    # Saatlik λ
    lam_hour = k_lambda * base_int[:, None] * diurnal[None, :]
    lam_hour *= (1.0 + near_repeat_alpha * nr[:, None])   # NR ile arttır
    lam_hour = np.clip(lam_hour, 0.0, 0.9)                # emniyet

    expected = lam_hour.sum(axis=1)
    p_hour = 1.0 - np.exp(-lam_hour)
    q10 = np.quantile(p_hour, 0.10, axis=1)
    q90 = np.quantile(p_hour, 0.90, axis=1)

    # Tür dağılımı
    rng = np.random.default_rng(42)
    alpha = np.array([1.5, 1.2, 2.0, 1.0, 1.3])
    W = rng.dirichlet(alpha, size=len(geo_df))
    types = expected[:, None] * W
    assault, burglary, theft, robbery, vandalism = types.T

    out = pd.DataFrame({
        KEY_COL: geo_df[KEY_COL].to_numpy(),
        "expected": expected,
        "q10": q10, "q90": q90,
        "assault": assault, "burglary": burglary, "theft": theft,
        "robbery": robbery, "vandalism": vandalism,
        "nr_boost": nr,     # ← NR skoru (0..1)
    })

    q90_thr = out["expected"].quantile(0.90)
    q70_thr = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90_thr, out["expected"] >= q70_thr],
        ["Yüksek", "Orta"], default="Hafif",
    )
    return out

# -------------------- Poisson yardımcıları --------------------
def p_to_lambda_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 0.999999)
    return -np.log1p(-p)

def p_to_lambda(p):
    p = np.clip(np.asarray(p, dtype=float), 0.0, 0.999999)
    return -np.log(1.0 - p)

def pois_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(k + 1):
        s += (lam ** i) / math.factorial(i)
    return math.exp(-lam) * s

def prob_ge_k(lam: float, k: int) -> float:
    return 1.0 - pois_cdf(k - 1, lam)

def pois_quantile(lam: float, q: float) -> int:
    k = 0
    while pois_cdf(k, lam) < q and k < 10_000:
        k += 1
    return k

def pois_pi90(lam: float) -> tuple[int, int]:
    lo = pois_quantile(lam, 0.05)
    hi = pois_quantile(lam, 0.95)
    return lo, hi
