# utils/geo.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import json
import numpy as np
import pandas as pd

from utils.constants import KEY_COL

def polygon_centroid(lonlat_loop):
    x, y = zip(*lonlat_loop)
    A = Cx = Cy = 0.0
    for i in range(len(lonlat_loop) - 1):
        cross = x[i]*y[i+1] - x[i+1]*y[i]
        A  += cross
        Cx += (x[i] + x[i+1]) * cross
        Cy += (y[i] + y[i+1]) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        return float(sum(x)/len(x)), float(sum(y)/len(y))
    return float(Cx/(6*A)), float(Cy/(6*A))

def load_geoid_layer(path="data/sf_cells.geojson", key_field=KEY_COL):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[key_field, "centroid_lon", "centroid_lat"]), []
    gj = json.loads(p.read_text(encoding="utf-8"))
    rows, feats_out = [], []
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        geoid = str(props.get(key_field) or props.get(key_field.upper())
                    or props.get("GEOID") or props.get("geoid") or "").strip()
        if not geoid:
            continue
        lon = props.get("centroid_lon")
        lat = props.get("centroid_lat")
        if lon is None or lat is None:
            geom = feat.get("geometry", {})
            if geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
            elif geom.get("type") == "MultiPolygon":
                ring = geom["coordinates"][0][0]
            else:
                continue
            lon, lat = polygon_centroid(ring)
        rows.append({key_field: geoid, "centroid_lon": float(lon), "centroid_lat": float(lat)})
        feat.setdefault("properties", {})["id"] = geoid
        feats_out.append(feat)
    return pd.DataFrame(rows), feats_out

def nearest_geoid(geo_df: pd.DataFrame, lat: float, lon: float) -> str | None:
    if geo_df.empty:
        return None
    la = geo_df["centroid_lat"].to_numpy()
    lo = geo_df["centroid_lon"].to_numpy()
    d2 = (la - lat) ** 2 + (lo - lon) ** 2
    i = int(np.argmin(d2))
    return str(geo_df.iloc[i][KEY_COL])

def _extract_latlon_from_ret(ret) -> Tuple[float, float] | None:
    if not ret:
        return None
    lc = ret.get("last_clicked")
    if lc is None:
        return None
    if isinstance(lc, (list, tuple)) and len(lc) >= 2:
        return float(lc[0]), float(lc[1])
    if isinstance(lc, dict):
        if "lat" in lc and ("lng" in lc or "lon" in lc):
            return float(lc["lat"]), float(lc.get("lng", lc.get("lon")))
        ll = lc.get("latlng")
        if isinstance(ll, (list, tuple)) and len(ll) >= 2:
            return float(ll[0]), float(ll[1])
        if isinstance(ll, dict) and "lat" in ll and ("lng" in ll or "lon" in ll):
            return float(ll["lat"]), float(ll.get("lng", ll.get("lon")))
    return None

def resolve_clicked_gid(geo_df: pd.DataFrame, ret: dict) -> tuple[str | None, tuple[float, float] | None]:
    gid, latlon = None, None
    obj = ret.get("last_object_clicked") if isinstance(ret, dict) else None
    if isinstance(obj, dict):
        props = obj.get("properties", {}) or obj.get("feature", {}).get("properties", {}) or {}
        gid = str(obj.get("id") or props.get("id") or props.get(KEY_COL)
                  or props.get("GEOID") or "").strip() or None
        if not latlon:
            geom = obj.get("geometry") or obj.get("feature", {}).get("geometry")
            if isinstance(geom, dict) and geom.get("type") == "Point":
                coords = geom.get("coordinates", [])
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    latlon = (float(coords[1]), float(coords[0]))
            if not latlon:
                lat = obj.get("lat") or obj.get("latlng", {}).get("lat") or obj.get("location", {}).get("lat")
                lng = obj.get("lng") or obj.get("latlng", {}).get("lng") or obj.get("location", {}).get("lng") or obj.get("lon")
                if lat is not None and lng is not None:
                    latlon = (float(lat), float(lng))
    if not gid:
        if not latlon:
            latlon = _extract_latlon_from_ret(ret)
        if latlon:
            gid = nearest_geoid(geo_df, latlon[0], latlon[1])
    return gid, latlon
