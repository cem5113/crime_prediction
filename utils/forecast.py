# utils/forecast.py
from __future__ import annotations
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.constants import CRIME_TYPES, KEY_COL


# -------------------- Baz yoğunluk: normalize edilmiş --------------------
def precompute_base_intensity(geo_df: pd.DataFrame) -> np.ndarray:
    """
    Hücre bazlı mekânsal yoğunluk (0..1 aralığına normalize).
    """
    lon = geo_df["centroid_lon"].to_numpy()
    lat = geo_df["centroid_lat"].to_numpy()

    peak1 = np.exp(-(((lon + 122.41) ** 2) / 0.0008 + ((lat - 37.78) ** 2) / 0.0005))
    peak2 = np.exp(-(((lon + 122.42) ** 2) / 0.0006 + ((lat - 37.76) ** 2) / 0.0006))
    raw = 0.2 + 0.8 * (peak1 + peak2) + 0.07

    raw = raw - raw.min()
    base = raw / (raw.max() + 1e-9)  # 0..1
    return base


# -------------------- Near-repeat yardımcıları --------------------
def _events_to_geoid(
    events: pd.DataFrame,
    geo_df: pd.DataFrame,
    max_age_hours: int = 72,
    now_dt: datetime | None = None,
) -> pd.Series:
    """
    Etkinlikleri en yakın hücreye (geoid) bağlar.
    - events: ts(datetime), lat, lon veya doğrudan 'geoid' içeriyor olabilir.
    Yalnızca son max_age_hours içindeki olaylar kullanılır.
    """
    if events is None or events.empty:
        return pd.Series([], dtype=str)

    ev = events.copy()

    # Zaman filtresi
    if "ts" in ev.columns and np.issubdtype(ev["ts"].dtype, np.datetime64):
        now_dt = now_dt or datetime.utcnow()
        cutoff = now_dt - timedelta(hours=max_age_hours)
        ev = ev.loc[ev["ts"] >= cutoff]
        if ev.empty:
            return pd.Series([], dtype=str)

    # Eğer zaten geoid var ise direkt onu kullan
    if KEY_COL in ev.columns:
        return ev[KEY_COL].astype(str)

    # lat/lon varsa en yakın centroid'e bağla
    needed = {"lat", "lon"}
    if needed.issubset(set(ev.columns)):
        la = geo_df["centroid_lat"].to_numpy()
        lo = geo_df["centroid_lon"].to_numpy()
        gids = geo_df[KEY_COL].astype(str).to_numpy()

        latv = ev["lat"].to_numpy(dtype=float)
        lonv = ev["lon"].to_numpy(dtype=float)

        # (N_events x N_cells) mesafe: vektörel yakınlaştırma
        # Büyük veri için KDTree tercih edilir; burada prototip basit yaklaşım.
        geoid_list = []
        for lt, ln in zip(latv, lonv):
            d2 = (la - lt) ** 2 + (lo - ln) ** 2
            i = int(np.argmin(d2))
            geoid_list.append(gids[i])
        return pd.Series(geoid_list, dtype=str)

    # ne geoid ne lat/lon yoksa boş
    return pd.Series([], dtype=str)


def _near_repeat_boost(
    geo_df: pd.DataFrame,
    recent_geoids: pd.Series,
    tau: float = 24.0,          # saatlik zaman ölçeği (azalma)
    spatial_sigma: float = 0.002,  # ~ mekânsal yayılım (yaklaşık derecelerde)
) -> np.ndarray:
    """
    Basit near-repeat skorlayıcı:
    - Her recent olay, çevredeki hücrelere Gauss mekânsal etki yayar (burada köşegen basitleştirilmiş).
    - Zaman etkisi _events_to_geoid içinde kesildiği için yalnızca mekânsal kernel kullanıyoruz.
    Çıktı 0..1 aralığına normalize edilir.
    """
    if recent_geoids.empty:
        return np.zeros(len(geo_df), dtype=float)

    # Hücre centroidleri
    la = geo_df["centroid_lat"].to_numpy()
    lo = geo_df["centroid_lon"].to_numpy()
    gids = geo_df[KEY_COL].astype(str).to_numpy()

    # Her hücre için skor = aynı geoid sayısı + komşu hücrelere Gauss yayılımı
    # Basitçe: olay gelen hücreye 1 puan verelim; komşulara mesafe bazlı azalarak
    # (Burada komşuyu bulmak yerine tüm hücrelere mesafe bazlı etki veriyoruz: prototip)
    # Etkinin toplamını 0..1 ölçeğine alacağız.

    # Olayların geldiği hücreleri index'e çevir
    gid_to_idx = {g: i for i, g in enumerate(gids)}
    hit_mask = np.zeros(len(geo_df), dtype=float)
    for g in recent_geoids:
        i = gid_to_idx.get(str(g))
        if i is not None:
            hit_mask[i] += 1.0

    # Mekânsal yayılım: her olaylı hücreden diğerlerine gauss( mesafe )
    score = np.zeros(len(geo_df), dtype=float)
    event_idx = np.where(hit_mask > 0)[0]
    for i in event_idx:
        # ilgili olayın sayısı kadar ağırlık
        w = hit_mask[i]
        d2 = (la - la[i]) ** 2 + (lo - lo[i]) ** 2
        score += w * np.exp(-d2 / (2 * spatial_sigma ** 2))

    # 0..1 normalizasyon
    if score.max() > 0:
        score = score / score.max()
    return score


# -------------------- Hızlı agregasyon: λ’yı kur + near-repeat (opsiyonel) --------------------
def aggregate_fast(
    start_iso: str,
    horizon_h: int,
    geo_df: pd.DataFrame,
    base_int: np.ndarray,
    *,
    k_lambda: float = 0.12,          # saatlik ölçek
    events: pd.DataFrame | None = None,
    near_repeat_alpha: float = 0.0,  # 0 → kapalı, 0.2–0.5 önerilir
) -> pd.DataFrame:
    """
    Saatlik λ = k_lambda * base_int * diurnal * (1 + near_repeat_alpha * nr_boost)
    Beklenen toplam = saatlik λ’ların toplamı.
    q10/q90: saatlik P(>=1)’in kantilleri.
    """
    start = datetime.fromisoformat(start_iso)
    hours = np.arange(horizon_h)
    diurnal = 1.0 + 0.4 * np.sin((((start.hour + hours) % 24 - 18) / 24) * 2 * np.pi)

    # Near-repeat skoru (0..1)
    if near_repeat_alpha > 0.0 and events is not None and not events.empty:
        recent_geoids = _events_to_geoid(events, geo_df, max_age_hours=72, now_dt=start)
        nr_boost = _near_repeat_boost(geo_df, recent_geoids, tau=24.0, spatial_sigma=0.002)
    else:
        nr_boost = np.zeros(len(geo_df), dtype=float)

    # Saatlik λ (etkinin çarpan olarak eklenmesi)
    mult = (1.0 + near_repeat_alpha * nr_boost).astype(float)
    lam_hour = np.clip(k_lambda * (base_int * mult)[:, None] * diurnal[None, :], 0.0, 0.9)

    # Toplam beklenen olay (λ toplamı)
    expected = lam_hour.sum(axis=1)

    # P(>=1) = 1 - e^{-λ}; belirsizlik için kantiller
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
        "nr_boost": nr_boost,  # ← UI’de göstereceğiz
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
