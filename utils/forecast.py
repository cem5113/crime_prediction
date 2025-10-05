# utils/forecast.py
from __future__ import annotations
import math
from datetime import datetime

import numpy as np
import pandas as pd

from utils.constants import CRIME_TYPES, KEY_COL


# -------------------- Baz yoğunluk: normalize edilmiş --------------------
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    """
    Hücre bazlı mekânsal yoğunluk (0..1 aralığına normalize).
    Doygunluğa girmemesi için min-max normalizasyonu uygulanır.
    """
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()

    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    raw = 0.2 + 0.8 * (peak1 + peak2) + 0.07  # önceki formül

    raw = raw - raw.min()
    base = raw / (raw.max() + 1e-9)          # 0..1
    return base


# -------------------- Hızlı agregasyon: λ’yı doğrudan kur --------------------
def aggregate_fast(
    start_iso: str,
    horizon_h: int,
    geo_df: pd.DataFrame,
    base_int: np.ndarray,
    *,
    k_lambda: float = 0.12,   # saatlik ölçek (gerekirse 0.08–0.15 arası ayarlayın)
) -> pd.DataFrame:
    """
    Saatlik λ = k_lambda * base_int * diurnal
    Beklenen toplam = saatlik λ’ların toplamı.
    q10/q90 belirsizlikleri saatlik P(>=1)’in kantilleriyle hesaplanır.
    """
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    # Saatlik λ (doygunluk yapma). Emniyet için üst sınır koyduk.
    lam_hour = np.clip(k_lambda * base_int[:, None] * diurnal[None, :], 0.0, 0.9)

    # Toplam beklenen olay (λ’ların toplamı)
    expected = lam_hour.sum(axis=1)

    # Saatlik P(>=1) = 1 - e^{-λ}; belirsizlik için kantiller
    p_hour = 1.0 - np.exp(-lam_hour)
    q10 = np.quantile(p_hour, 0.10, axis=1)
    q90 = np.quantile(p_hour, 0.90, axis=1)

    # Tür kırılımı
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

    # Öncelik sınıfları
    q90_thr = out["expected"].quantile(0.90)
    q70_thr = out["expected"].quantile(0.70)
    out["tier"] = np.select(
        [out["expected"] >= q90_thr, out["expected"] >= q70_thr],
        ["Yüksek", "Orta"], default="Hafif",
    )
    return out


# -------------------- Poisson yardımcıları (kart/tablolar için) --------------------
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
