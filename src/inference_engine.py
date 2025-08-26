## 4) src/inference_engine.py

```python
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional
from joblib import load
from .config import paths, params
from .common import ensure_dir, clean_geoid, infer_hour_from_col, to_hour_range, safe_read_csv, minmax01
from .features import _recent_trend, neighbor_spillover, load_centroids, crime_mix_prior

class DummyPredictor:
    """Model yoksa UI'nin çalışması için basit bir tahminleyici.
    Saat, son olay yoğunluğu ve komşuluk sinyalinden basit olasılık türetir.
    """
    def predict_proba(self, X: pd.DataFrame):
        rng = np.random.default_rng(42)
        base = 0.08 + 0.02 * (X.get("hour", pd.Series([0]*len(X))).values/23.0)
        bump = 0.05 * np.tanh(np.nan_to_num(X.get("trend_z", pd.Series([0]*len(X))).values))
        neigh = 0.03 * np.tanh(np.nan_to_num(X.get("neighbor_spillover", pd.Series([0]*len(X))).values)/5)
        p = np.clip(base + bump + neigh + rng.normal(0, 0.01, size=len(X)), 0.001, 0.95)
        return np.vstack([1-p, p]).T

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.calibrator = None
        self.residuals = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            if os.path.isfile(paths.STACK_MODEL):
                self.model = load(paths.STACK_MODEL)
        except Exception:
            self.model = None
        try:
            if os.path.isfile(paths.CALIBRATOR):
                self.calibrator = load(paths.CALIBRATOR)
        except Exception:
            self.calibrator = None
        try:
            if os.path.isfile(paths.CONFORMAL_RESIDUALS):
                self.residuals = np.load(paths.CONFORMAL_RESIDUALS)
        except Exception:
            self.residuals = None
        if self.model is None:
            self.model = DummyPredictor()

    def _calibrate(self, p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return p
        try:
            return self.calibrator.predict_proba(p.reshape(-1,1))[:,1]
        except Exception:
            return p

    def _conformal_band(self, p: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        # Basit konformal genişlik: rezidüellerin (quantile) ile +- aralık
        if self.residuals is None or len(self.residuals) < 50:
            width = 0.07  # varsayılan bant
            return np.clip(p - width, 0, 1), np.clip(p + width, 0, 1)
        lo = np.quantile(self.residuals, 1-alpha)
        return np.clip(p - lo, 0, 1), np.clip(p + lo, 0, 1)

    def _priority(self, p_cal: np.ndarray, trend_z: np.ndarray, persistence: np.ndarray, neigh: np.ndarray, route_cost: np.ndarray) -> np.ndarray:
        # Priority skor (0-100) — lineer harman, ardından 0-100 ölçek
        w = [0.6, 0.15, 0.10, 0.10, -0.10]
        raw = w[0]*p_cal + w[1]*trend_z + w[2]*persistence + w[3]*neigh + w[4]*route_cost
        mm = np.array(minmax01(raw))
        return (100*mm).astype(float)

    def _route_cost_proxy(self, centroids: Optional[pd.DataFrame], geoids: list[str]) -> np.ndarray:
        # Basit: şehir merkezine (lat/lon medyan) uzaklık normalizasyonu (düşük=iyi)
        if centroids is None or len(geoids)==0:
            return np.zeros(len(geoids), dtype=float)
        C = centroids.set_index("GEOID")
        sub = C.loc[C.index.intersection(geoids)]
        if sub.empty:
            return np.zeros(len(geoids), dtype=float)
        lat0, lon0 = sub["lat"].median(), sub["lon"].median()
        d = np.sqrt((sub["lat"]-lat0)**2 + (sub["lon"]-lon0)**2).reindex(geoids).fillna(d.max() if len(d)>0 else 0)
        d = d.values
        if d.max() - d.min() < 1e-9:
            return np.zeros_like(d)
        return (d - d.min())/(d.max()-d.min())

    def predict_topk(self, hour_label: Optional[str] = None, topk: Optional[int] = None) -> pd.DataFrame:
        topk = topk or params.TOP_K
        # GEOID listesi ve saat üretimi için sf_crime_50 veya 52 veya crime
        sf50 = safe_read_csv(paths.SF50_CSV)
        sf52 = safe_read_csv(paths.SF52_CSV)
        sfcrime = safe_read_csv(paths.SF_CRIME_CSV)
        centroids = load_centroids()

        # GEOID evreni
        geoids = None
        if sf50 is not None and "GEOID" in sf50.columns:
            geoids = sf50["GEOID"].astype(str).unique().tolist()
        elif sf52 is not None and "GEOID" in sf52.columns:
            geoids = sf52["GEOID"].astype(str).unique().tolist()
        elif sfcrime is not None and "GEOID" in sfcrime.columns:
            geoids = sfcrime["GEOID"].astype(str).unique().tolist()
        else:
            # örnek GEOID üret (sunum güvenliği için)
            geoids = [f"{i:011d}" for i in range(60750100000, 60750100000+200)]

        geoids = [str(g)[:params.GEOID_LEN].zfill(params.GEOID_LEN) for g in geoids]
        hour_label = hour_label or to_hour_range(20, params.HOUR_BIN)  # varsayılan 20-22

        rows = []
        for geoid in geoids:
            # trend & persistence
            tz, pers = _recent_trend(sfcrime, geoid, hour_label, days=14)
            # komşuluk
            spill = neighbor_spillover(centroids, geoid, k=5)
            rows.append({
                "GEOID": geoid,
                "hour_range": hour_label,
                "trend_z": tz,
                "persistence": pers,
                "neighbor_spillover": spill,
            })
        X = pd.DataFrame(rows)
        # özellik setini minimal tut — model yoksa DummyPredictor zaten tolere eder
        X["hour"] = int(hour_label.split(":")[0]) if isinstance(hour_label, str) else 20

        # p_crime
        proba = self.model.predict_proba(X)[:,1]
        p_cal = self._calibrate(proba)
        lcb, ucb = self._conformal_band(p_cal, alpha=0.1)

        # route cost proxy
        rcost = self._route_cost_proxy(centroids, [r["GEOID"] for r in rows])
        priority = self._priority(p_cal, X["trend_z"].fillna(0).values, X["persistence"].fillna(0).values, X["neighbor_spillover"].fillna(0).values, rcost)

        # crime mix
        mix_str = []
        for geoid in geoids:
            mix_str.append(crime_mix_prior(sf50, geoid, hour_label, topn=3))

        out = X.copy()
        out["p_crime"] = p_cal
        out["lcb"], out["ucb"] = lcb, ucb
        out["priority_score"] = priority
        out["top3_crime_types"] = mix_str
        out = out.sort_values(["priority_score","p_crime"], ascending=False).head(topk)
        out.insert(0, "rank", range(1, len(out)+1))
        return out.reset_index(drop=True)

    def save_topk(self, df: pd.DataFrame) -> str:
        ensure_dir(paths.RISK_DIR)
        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d_%H")
        outpath = os.path.join(paths.RISK_DIR, f"topk_geoid_{ts}.csv")
        df.to_csv(outpath, index=False)
        return outpath
