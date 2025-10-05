# utils/patrol.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from utils.constants import KEY_COL

def kmeans_like(coords: np.ndarray, weights: np.ndarray, k: int, iters: int = 20):
    n = len(coords); k = min(k, n)
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

def allocate_patrols(df_agg: pd.DataFrame, geo_df: pd.DataFrame,
                     k_planned: int, duty_minutes: int,
                     cell_minutes: int = 6, travel_overhead: float = 0.40) -> Dict:
    cand = df_agg[df_agg["tier"].isin(["Yüksek", "Orta"])].copy()
    if cand.empty:
        return {"zones": []}
    merged  = cand.merge(geo_df, on=KEY_COL)
    coords  = merged[["centroid_lon", "centroid_lat"]].to_numpy()
    weights = merged["expected"].to_numpy()
    k = max(1, min(int(k_planned), 50))
    cents, assign = kmeans_like(coords, weights, k)

    cap_cells = max(1, int(duty_minutes / (cell_minutes * (1.0 + travel_overhead))))
    zones = []
    for z in range(len(cents)):
        m = assign == z
        if not np.any(m):
            continue
        sub = merged[m].copy().sort_values("expected", ascending=False)
        sub_planned = sub.head(cap_cells).copy()
        cz = cents[z]
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
