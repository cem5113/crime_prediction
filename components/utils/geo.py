# components/utils/geo.py
from __future__ import annotations
import json, os
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Shapely opsiyonel (varsa centroid daha doğru hesaplanır)
try:
    from shapely.geometry import shape as shp_shape
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

# Paket içi sabit
try:
    from .constants import KEY_COL as DEFAULT_KEY_COL
except Exception:
    DEFAULT_KEY_COL = "geoid"


# --------------------------- yardımcılar ---------------------------

def _infer_key(props: Dict[str, Any]) -> Optional[str]:
    """properties içinden GEOID değerini tekilleştir."""
    for k in ("geoid", "GEOID", "id", "Id", "ID"):
        if k in props and props[k] not in (None, ""):
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
            return (float(c.x), float(c.y))  # (lon, lat)
        except Exception:
            pass

    # Basit fallback: yalnızca dış halka ortalaması
    try:
        if gtype == "Polygon" and coords and len(coords) > 0:
            ring = coords[0]  # dış halka
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        elif gtype == "MultiPolygon" and coords and len(coords) > 0:
            ring = coords[0][0]
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        elif gtype == "Point" and len(coords) == 2:
            return (float(coords[0]), float(coords[1]))
    except Exception:
        return None
    return None


# --------------------------- yükleyiciler ---------------------------

def load_geoid_layer(path: str, key_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Tek bir dosya yolundan grid yükler.
    Dönüş: (geo_df, geo_features)
      geo_df: [key_col, centroid_lat, centroid_lon, neighborhood?]
      geo_features: GeoJSON features list (orijinal)
    """
    return _load_geoid_layer_any([path], key_col=key_col)[0:2]


def load_geoid_layer_any(
    candidates: List[str],
    key_col: Optional[str] = None,
    return_debug: bool = False
):
    """
    Birden fazla yoldan ilk başarılı grid’i yükle.
    Dönüş:
      - return_debug=False → (geo_df, geo_features)
      - return_debug=True  → (geo_df, geo_features, debug_lines)
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
                    # properties içinde belirgin anahtar yoksa bu feature'ı geç
                    continue

                # mahalle adı varsa al
                nbhd = None
                for ncand in ("neighborhood", "neighbourhood", "name", "NAME"):
                    if ncand in props and props[ncand]:
                        nbhd = props[ncand]
                        break

                # centroid_lat/lon varsa doğrudan kullan, yoksa hesapla
                lat = props.get("centroid_lat")
                lon = props.get("centroid_lon")
                if lat is None or lon is None:
                    cen = _centroid_from_geometry(geom)
                    if cen is not None:
                        lon, lat = cen[0], cen[1]

                rows.append({
                    kcol: str(key_val),
                    "centroid_lat": float(lat) if lat is not None else None,
                    "centroid_lon": float(lon) if lon is not None else None,
                    "neighborhood": nbhd,
                })

            df = pd.DataFrame(rows).drop_duplicates(subset=[kcol])

            if df.empty:
                debug_lines.append("  → satır toplanamadı (anahtar/centroid bulunamadı).")
                continue

            # centroid sütunlarını garanti et
            for c in ("centroid_lat", "centroid_lon"):
                if c not in df.columns:
                    df[c] = None

            debug_lines.append(f"  → başarı: {len(df)} satır")
            return df.reset_index(drop=True), feats, debug_lines

        except Exception as e:
            last_err = e
            debug_lines.append(f"  → hata: {e}")

    # Hepsi başarısız olduysa boş dön
    if last_err:
        debug_lines.append(f"Son hata: {last_err}")
    return pd.DataFrame(columns=[kcol, "centroid_lat", "centroid_lon", "neighborhood"]), [], debug_lines


# --------------------------- folium tıklama çözücü ---------------------------

def resolve_clicked_gid(
    geo_df: pd.DataFrame,
    folium_return: dict,
    key_col: Optional[str] = None
) -> tuple[Optional[str], str]:
    """
    st_folium(...) dönüşünden GEOID çözer.
    1) last_object_clicked varsa → properties.id/geoID al.
    2) Aksi halde last_clicked (lat/lon) → en yakın centroid.
    Dönüş: (geoid_str | None, method_str)
    """
    kcol = key_col or DEFAULT_KEY_COL
    if not isinstance(folium_return, dict):
        return None, "invalid_ret"

    # 1) Polygon tıklaması (GeoJsonTooltip/Popup ile)
    loc = folium_return.get("last_object_clicked")
    if isinstance(loc, dict):
        props = loc.get("properties") or {}
        gid = props.get("id") or props.get("geoid") or props.get("GEOID")
        if gid is not None:
            return str(gid), "feature_id"

    # 2) Harita tıklaması (koordinata en yakın hücre centroid’i)
    last = folium_return.get("last_clicked")
    if isinstance(last, dict) and "lat" in last and "lng" in last:
        lat = float(last["lat"]); lon = float(last["lng"])
        if {"centroid_lat", "centroid_lon"}.issubset(geo_df.columns) and not geo_df.empty:
            arr = geo_df[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float)
            d2  = (arr[:, 0] - lat) ** 2 + (arr[:, 1] - lon) ** 2
            idx = int(np.argmin(d2))
            gid = geo_df.iloc[idx][kcol]
            return str(gid), "nearest_centroid"

    return None, "not_found"
