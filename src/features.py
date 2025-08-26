## 3) src/features.py

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .config import paths, params
from .common import clean_geoid, infer_hour_from_col, to_hour_range, safe_read_csv

# ——— Trend ve kalıcılık (persistence) yaklaşıkları ———
# Veri yoksa 0 döner; varsa Z-skoru ve persistence üretir.

def _recent_trend(df: pd.DataFrame, geoid: str, hlabel: str, days: int = 14) -> Tuple[float, float]:
    try:
        # sf_crime.csv varsa son 14 gün aynı saat aralığı olay sayısı
        if df is None or df.empty:
            return 0.0, 0.0
        d = df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            d = d[d["date"] >= (d["date"].max() - pd.Timedelta(days=days))]
        # hour_range üretimi
        hr = infer_hour_from_col(d)
        d["hour_range"] = hr.apply(lambda h: to_hour_range(h, params.HOUR_BIN))
        d["GEOID"] = clean_geoid(d["GEOID"], params.GEOID_LEN)
        dd = d[(d["GEOID"]==geoid) & (d["hour_range"]==hlabel)]
        if dd.empty:
            return 0.0, 0.0
        # Gün bazında say — basit trend z ve persistence (son 3 günün ort/öncesine oran)
        g = dd.groupby(dd["date"].dt.date).size().astype(float)
        if len(g) < 5:
            return 0.0, 0.0
        z = (g.diff().fillna(0).mean()) / (g.std()+1e-9)
        recent = g.tail(3).mean()
        older = g.iloc[:-3].mean() if len(g) > 3 else g.mean()
        persistence = float((recent - older) / (older + 1e-6))
        return float(z), persistence
    except Exception:
        return 0.0, 0.0

# ——— Komşuluk taşıması (kNN centroid üzerinden) ———

def neighbor_spillover(centroids: pd.DataFrame, target_geoid: str, k: int = 5) -> float:
    try:
        from sklearn.neighbors import BallTree
        if centroids is None or centroids.empty:
            return 0.0
        row = centroids.loc[centroids["GEOID"]==target_geoid]
        if row.empty:
            return 0.0
        lat, lon = float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"])
        # BallTree için radyan
        R = np.pi/180.0
        pts = np.vstack([centroids["lat"].values*R, centroids["lon"].values*R]).T
        tree = BallTree(pts, metric="haversine")
        dist, ind = tree.query(np.array([[lat*R, lon*R]]), k=min(k+1, len(centroids)))
        # ilk index kendisi; komşuların uzaklık ters ağırlıklı katsayısı
        w = 0.0
        for j, d in zip(ind[0][1:], dist[0][1:]):
            w += 1.0/float(d+1e-6)
        return float(w / (k if k>0 else 1))
    except Exception:
        return 0.0

# ——— GEOID centroid okuma/üretme ———

def load_centroids() -> Optional[pd.DataFrame]:
    c = safe_read_csv(paths.TRACT_CENTROIDS_CSV)
    if c is not None and {"GEOID","lat","lon"}.issubset(c.columns):
        c["GEOID"] = clean_geoid(c["GEOID"], params.GEOID_LEN)
        return c[["GEOID","lat","lon"]].dropna()
    # geojson'dan üret
    try:
        import geopandas as gpd
        if os.path.isfile(paths.TRACTS_GEOJSON):
            g = gpd.read_file(paths.TRACTS_GEOJSON)
            if "GEOID" in g.columns:
                g["GEOID"] = g["GEOID"].astype(str).str.zfill(params.GEOID_LEN)
                g["centroid"] = g.geometry.centroid
                c = pd.DataFrame({
                    "GEOID": g["GEOID"].values,
                    "lat": g["centroid"].y.values,
                    "lon": g["centroid"].x.values,
                })
                return c
    except Exception:
        pass
    return None

# ——— Crime-mix (tahmini suç dağılımı) ———

def crime_mix_prior(sf50: Optional[pd.DataFrame], geoid: str, hour_label: str, topn: int = 5) -> str:
    try:
        if sf50 is None or sf50.empty:
            return ""
        cols = [c for c in sf50.columns if c.lower().startswith("type_")]
        if not cols:
            return ""
        dd = sf50.copy()
        if "GEOID" in dd.columns:
            dd["GEOID"] = clean_geoid(dd["GEOID"], params.GEOID_LEN)
        if "hour_range" in dd.columns:
            mask = (dd["GEOID"]==geoid) & (dd["hour_range"]==hour_label)
        else:
            mask = (dd["GEOID"]==geoid)
        dd = dd.loc[mask, cols].mean().sort_values(ascending=False)
        dd = dd.head(min(topn, len(dd)))
        parts = [f"{k.replace('type_','').title()}({v:.0%})" for k,v in dd.items()]
        return ", ".join(parts)
    except Exception:
        return ""
