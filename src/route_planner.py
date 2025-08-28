## 5) src/route_planner.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict
from .features import load_centroids
from .config import params

# Basit: top noktaları sırala, ardından 2-opt ile küçük iyileştirme

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi/180
    dlat = (lat2-lat1)*p
    dlon = (lon2-lon1)*p
    a = np.sin(dlat/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R*c

def _route_length(lats, lons, order):
    d = 0.0
    for i in range(len(order)-1):
        a, b = order[i], order[i+1]
        d += _haversine(lats[a], lons[a], lats[b], lons[b])
    return d

def two_opt(order, lats, lons, iters=50):
    best = order.copy()
    best_len = _route_length(lats, lons, best)
    for _ in range(iters):
        i, j = sorted(np.random.choice(range(1, len(order)-1), 2, replace=False))
        cand = best[:i] + best[i:j][::-1] + best[j:]
        cand_len = _route_length(lats, lons, cand)
        if cand_len < best_len:
            best, best_len = cand, cand_len
    return best

def plan_routes(topk_df: pd.DataFrame, n_teams: int = None, time_budget_min: int = None) -> List[Dict]:
    n_teams = n_teams or params.N_TEAMS
    time_budget_min = time_budget_min or params.TIME_BUDGET_MIN
    cent = load_centroids()
    if cent is None or cent.empty or topk_df.empty:
        return []
    M = topk_df.merge(cent, on="GEOID", how="left")
    M = M.dropna(subset=["lat","lon"]).reset_index(drop=True)

    # basit bölüşüm: sırayı n_teams'e böl
    chunks = np.array_split(M.index.tolist(), n_teams)
    routes = []
    for ridx, idxs in enumerate(chunks, start=1):
        lats = M.loc[idxs, "lat"].values
        lons = M.loc[idxs, "lon"].values
        base_order = list(range(len(idxs)))
        order = two_opt(base_order, lats, lons, iters=60)
        ordered = M.loc[idxs].iloc[order]
        dist = _route_length(ordered["lat"].values, ordered["lon"].values, list(range(len(ordered))))
        routes.append({
            "route_id": f"R{ridx}",
            "ordered_geoids": ordered["GEOID"].tolist(),
            "distance_km": float(dist),
            "eta_min": float(dist/20*60),  # 20 km/sa varsayım
            "expected_coverage_gain": float(ordered["priority_score"].sum()/100.0),
            "fairness_gap": 0.0,
        })
    return routes
