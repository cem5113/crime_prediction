# components/utils/geo.py
from __future__ import annotations
import json, os
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd

try:
    # Shapely varsa poligon centroid hesabı çok daha doğru olur
    from shapely.geometry import shape as shp_shape
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

# Paket-içi sabit
try:
    from .constants import KEY_COL as DEFAULT_KEY_COL
except Exception:
    DEFAULT_KEY_COL = "geoid"


def _infer_key(props: Dict[str, Any]) -> Optional[str]:
    """properties içinden GEOID değerini tekilleştir."""
    for k in ("geoid", "GEOID", "id", "Id", "ID"):
        if k in props and props[k] is not None:
            return str(props[k])
    return None


def _centroid_from_geometry(geom: Dict[str, Any]) -> Optional[tuple]:
    """GeoJSON geometry'den (lon, lat) centroid üret."""
    if not geom:
        return None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not gtype or coords is None:
        return None

    # Shapely varsa en doğrusu:
    if SHAPELY_OK:
        try:
            g = shp_shape(geom)
            c = g.centroid
            return (float(c.x), float(c.y))
        except Exception:
            pass

    # Basit fallback: yalnızca Polygon dış halkadan kaba centroid
    try:
        if gtype == "Polygon" and coords and len(coords) > 0:
            ring = coords[0]  # dış halka
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        elif gtype == "MultiPolygon" and coords and len(coords) > 0:
            # ilk poligonun dış halkası
            ring = coords[0][0]
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        elif gtype == "Point" and coords and len(coords) == 2:
            return (float(coords[0]), float(coords[1]))
    except Exception:
        return None
    return None


def load_geoid_layer(path: str, key_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Tek bir dosya yolundan grid yükler.
    Dönüş: (geo_df, geo_features)
      geo_df: [key_col, centroid_lat, centroid_lon, neighborhood?]
      geo_features: GeoJSON features list (orijinal)
    """
    debug = []  # iç teşhis için (kullanmasan da olur)
    return _load_geoid_layer_any([path], key_col=key_col)[0:2]


def load_geoid_layer_any(candidates: List[str], key_col: Optional[str] = None, return_debug: bool = False):
    """
    Birden fazla yoldan ilk başarılı grid’i yükle.
    Dönüş: (geo_df, geo_features, debug_lines?)  # return_debug=True ise 3. eleman döner
    """
    geo_df, features, debug = _load_geoid_layer_any(candidates, key_col=key_col)
    if return_debug:
        return geo_df, features, debug
    return geo_df, features


def _load_geoid_layer_any(candidates: List[str], key_col: Optional[str] = None):
    kcol = key_col or DEFAULT_KEY_COL
    debug_lines: List[str] = []
    last_err = None

    for p in candidates:
        if not p:
            continue
        try:
            debug_lines.append(f"Deniyorum: {p}")
            if not os.path.exists(p):
                debug_lines.append("  → yol bulunamadı.")
                continue
            size = os.path.getsize(p)
            debug_lines.append(f"  → boyut: {size} bayt")
            if size <= 10:
                debug_lines.append("  → dosya çok küçük/boş görünüyor.")
                continue

            with open(p, "r", encoding="utf-8") as f:
                js = json.load(f)

            feats = js.get("features", [])
            debug_lines.append(f"  → feature sayısı: {len(feats)}")
            if not feats:
                continue

            rows = []
            for ft in feats:
                props = ft.get("properties", {}) or {}
                geom = ft.get("geometry", {}) or {}

                key_val = _infer_key(props)
                if key_val is None:
                    # properties içinde belirgin anahtar yoksa path’i eliyoruz
                    continue

                # mahalle adı varsa al
                nbhd = None
                for ncand in ("neighborhood", "neighbourhood", "name", "NAME"):
                    if ncand in props:
                        nbhd = props[ncand]
                        break

                # centroid_lat/lon varsa doğrudan kullan, yoksa hesapla
                lat = props.get("centroid_lat")
                lon = props.get("centroid_lon")
                if lat is None or lon is None:
                    cen = _centroid_from_geometry(geom)
                    if cen is not None:
                        lon, lat = cen[0], cen[1]

                if lat is None or lon is None:
                    # yine de bırak ama sonraki aşamada filtrelenebilir
                    lat = None
                    lon = None

                rows.append({
                    kcol: str(key_val),
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "neighborhood": nbhd,
                })

            df = pd.DataFrame(rows).drop_duplicates(subset=[kcol])
            # Satır yoksa bu aday başarısız sayılır
            if df.empty:
                debug_lines.append("  → satır toplanamadı (anahtar/centroid bulunamadı).")
                continue

            # centroid yoksa en azından sütunlar mevcut olsun
            if "centroid_lat" not in df.columns:
                df["centroid_lat"] = None
            if "centroid_lon" not in df.columns:
                df["centroid_lon"] = None

            debug_lines.append(f"  → başarı: {len(df)} satır")
            return df.reset_index(drop=True), feats, debug_lines

        except Exception as e:
            last_err = e
            debug_lines.append(f"  → hata: {e}")

    # Hepsi başarısız olduysa boş dön
    if last_err:
        debug_lines.append(f"Son hata: {last_err}")
    return pd.DataFrame(columns=[kcol, "centroid_lat", "centroid_lon", "neighborhood"]), [], debug_lines
