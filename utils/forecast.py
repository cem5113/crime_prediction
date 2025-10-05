# utils/forecast.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from datetime import datetime
from utils.constants import CRIME_TYPES, KEY_COL

def p_to_lambda_array(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, 0.999999)
    return -np.log1p(-p)

def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()
    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    noise = 0.07
    return 0.2 + 0.8 * (peak1 + peak2) + noise

def aggregate_fast(start_iso: str, horizon_h: int, geo_df: pd.DataFrame, base_int: np.ndarray) -> pd.DataFrame:
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    p = np.clip(base_int[:, None] * diurnal[None, :], 0, 1)
    p_any = np.clip(0.05 + 0.5 * p, 0, 0.98)

    lam = p_to_lambda_array(p_any)
    expected = lam.sum(axis=1)
    q10 = np.maximum(0.0, p_any - 0.08).mean(axis=1)
    q90 = np.minimum(1.0, p_any + 0.08).mean(axis=1)

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
    })

    q90_thr = out["expected"].quantile(0.90)
    q70_thr = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90_thr, out["expected"] >= q70_thr],
        ["Yüksek", "Orta"], default="Hafif",
    )
    return out

# --- Poisson yardımcıları (kart için) ---
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
