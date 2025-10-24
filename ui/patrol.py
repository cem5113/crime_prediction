# components/utils/patrol.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Paket içi sabit (göreli import)
try:
    from .constants import KEY_COL as DEFAULT_KEY_COL
except Exception:
    DEFAULT_KEY_COL = "geoid"


@dataclass
class PatrolParams:
    k_planned: int = 6
    duty_minutes: int = 120
    cell_minutes: int = 6
    travel_overhead: float = 0.40  # toplam süreye oransal ek


def _kmeans_like(coords: np.ndarray, weights: np.ndarray, k: int, iters: int = 20):
    """
    Ağırlıklı k-means benzeri çok hafif kümeleme (merkezleri risk ağırlıklarıyla günceller).
    """
    n = len(coords)
    k = max(1, min(k, n))
    if n == 0:
        return np.empty((0, 2)), np.empty((0,), dtype=int)

    # en ağır k noktayı başlangıç merkezi yap
    idx_sorted = np.argsort(-weights)
    centroids = coords[idx_sorted[:k]].astype(float).copy()
    assign = np.zeros(n, dtype=int)

    for _ in range(max(iters, 1)):
        # atama
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        assign = np.argmin(dists, axis=1)
        # merkezleri güncelle
        for c in range(k):
            m = assign == c
            if not np.any(m):
                # boş küme: en ağır noktaya sıçra
                centroids[c] = coords[idx_sorted[0]]
            else:
                w = weights[m][:, None]
                centroids[c] = (coords[m] * w).sum(axis=0) / max(1e-6, w.sum())

    return centroids, assign


def allocate_patrols(
    df_agg: pd.DataFrame,
    geo_df: pd.DataFrame,
    *,
    k_planned: int,
    duty_minutes: int,
    cell_minutes: int = 6,
    travel_overhead: float = 0.40,
    key_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    app.py'nin çağırdığı planlayıcı.
    En riskli hücreleri K devriyeye paylaştırır ve app.py'nin beklediği yapıyı döner:

    return {
      "zones": [
        {
          "id": "Z1",
          "centroid": {"lat": ..., "lon": ...},
          "cells": [<geoid,...>],
          "route": [[lat,lon], ...],
          "expected_risk": float,
          "planned_cells": int,
          "eta_minutes": int,
          "utilization_pct": int,
          "capacity_cells": int,
        }, ...
      ]
    }
    """
    kcol = key_col or DEFAULT_KEY_COL

    # Gerekli kolon kontrolleri
    need_cols_agg = {kcol, "expected", "tier"}
    need_cols_geo = {kcol, "centroid_lat", "centroid_lon"}
    for c in need_cols_agg:
        if c not in df_agg.columns:
            raise KeyError(f"allocate_patrols: '{c}' sütunu agg_df'te yok.")
    for c in need_cols_geo:
        if c not in geo_df.columns:
            raise KeyError(f"allocate_patrols: '{c}' sütunu geo_df'te yok.")

    # Adaylar: Yüksek + Orta
    cand = df_agg[df_agg["tier"].isin(["Yüksek", "Orta"])].copy()
    if cand.empty:
        return {"zones": []}

    merged = cand[[kcol, "expected"]].merge(
        geo_df[[kcol, "centroid_lat", "centroid_lon"]],
        on=kcol, how="left"
    ).dropna(subset=["centroid_lat", "centroid_lon"])

    if merged.empty:
        return {"zones": []}

    coords = merged[["centroid_lon", "centroid_lat"]].to_numpy(float)  # (x=lon, y=lat)
    weights = merged["expected"].to_numpy(float)

    K = max(1, min(int(k_planned), 50))
    centroids, assign = _kmeans_like(coords, weights, K)

    # Kapasite (hücre adedi) — yol payı dahil
    capacity_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + float(travel_overhead)))))

    zones: List[Dict[str, Any]] = []
    for z in range(len(centroids)):
        m = assign == z
        if not np.any(m):
            continue

        sub = merged[m].copy().sort_values("expected", ascending=False)
        sub_planned = sub.head(capacity_cells).copy()

        cz_lon, cz_lat = centroids[z]
        # kaba "rota" sıralaması: merkez açısına göre
        angles = np.arctan2(sub_planned["centroid_lat"] - cz_lat, sub_planned["centroid_lon"] - cz_lon)
        sub_planned = sub_planned.assign(angle=angles).sort_values("angle")

        route = sub_planned[["centroid_lat", "centroid_lon"]].to_numpy().tolist()
        n_cells = len(sub_planned)
        eta_minutes = int(round(n_cells * cell_minutes * (1.0 + float(travel_overhead))))
        util = min(100, int(round(100.0 * n_cells / max(1, capacity_cells))))

        zones.append({
            "id": f"Z{z+1}",
            "centroid": {"lat": float(cz_lat), "lon": float(cz_lon)},
            "cells": sub_planned[kcol].astype(str).tolist(),
            "route": route,
            "expected_risk": float(sub_planned["expected"].mean() if n_cells else 0.0),
            "planned_cells": int(n_cells),
            "eta_minutes": int(eta_minutes),
            "utilization_pct": int(util),
            "capacity_cells": int(capacity_cells),
        })

    return {"zones": zones}
