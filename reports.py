# utils/reports.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- Dahili yardımcılar ----

_TS_CANDIDATES = [
    "ts", "timestamp", "datetime", "date_time", "reported_at", "occurred_at",
    "time", "event_time"
]

_LAT_SYNONYMS = ["lat", "latitude", "y", "Lat", "Latitude"]
_LON_SYNONYMS = ["lon", "long", "longitude", "x", "Lon", "Long", "Longitude"]

def _pick_first_existing(lower_map: dict[str, str], names: list[str]) -> str | None:
    for n in names:
        if n in lower_map:
            return lower_map[n]
    return None

def _normalize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Lat/Lon eşanlamlılarını 'latitude'/'longitude' olarak normalize eder."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    lower = {c.lower(): c for c in out.columns}

    lat_col = _pick_first_existing(lower, [s.lower() for s in _LAT_SYNONYMS])
    lon_col = _pick_first_existing(lower, [s.lower() for s in _LON_SYNONYMS])

    if lat_col and "latitude" not in out.columns:
        out.rename(columns={lower[lat_col]: "latitude"}, inplace=True)
    if lon_col and "longitude" not in out.columns:
        out.rename(columns={lower[lon_col]: "longitude"}, inplace=True)

    # Tipleri güvene al
    if "latitude" in out.columns:
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    if "longitude" in out.columns:
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")

    return out

def _infer_ts_series(df: pd.DataFrame) -> pd.Series | None:
    """
    Zaman bilgisini şu öncelikle türetir:
    1) Bilinen aday kolonlardan biri
    2) Ayrı 'date' + 'time'
    3) Yoksa None (rapor tarafı uyarı verir)
    """
    lower = {c.lower(): c for c in df.columns}

    # 1) Aday kolonlardan biri
    found = _pick_first_existing(lower, [s.lower() for s in _TS_CANDIDATES])
    if found:
        s = df[lower[found]]
        if np.issubdtype(s.dtype, np.number):
            med = pd.Series(s).dropna().astype(float).median()
            unit = "ms" if med and med > 1e12 else "s"
            ts = pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(s.astype(str).str.strip(), utc=True, errors="coerce")
        return ts

    # 2) date + time birleşik
    if "date" in lower and "time" in lower:
        tmp = df[[lower["date"], lower["time"]]].astype(str).agg(" ".join, axis=1)
        ts = pd.to_datetime(tmp, utc=True, errors="coerce")
        return ts

    # 3) YOK
    return None

def _validate_min_schema(df: pd.DataFrame, key_col: str) -> bool:
    """En azından geoid ya da (latitude, longitude) var mı?"""
    has_key = key_col in df.columns
    has_ll  = {"latitude", "longitude"} <= set(df.columns)
    return bool(has_key or has_ll)

# ---- Dışa açık yardımcılar ----

def normalize_events_ts(df: pd.DataFrame, key_col: str = "geoid") -> pd.DataFrame:
    """
    Verilen DataFrame'de zaman bilgisini 'ts' kolonu olarak türetir ve UTC'ye çevirir.
    Ek olarak 'hour', 'dow', 'date' kolonlarını ekler ve lat/lon isimlerini normalize eder.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = _normalize_latlon(df)

    ts = _infer_ts_series(out)
    if ts is None:
        # Belki zaten 'ts' vardır ama parsable değildir; tekrar dene
        if "ts" in out.columns:
            ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        else:
            return pd.DataFrame()

    out = out.copy()
    out["ts"] = ts
    out = out.dropna(subset=["ts"])
    if out.empty:
        return pd.DataFrame()

    out["hour"] = out["ts"].dt.hour
    out["dow"]  = out["ts"].dt.dayofweek
    out["date"] = out["ts"].dt.date

    # Minimum şema kontrol (raporlar yine çalışır; sadece bazı UI parçaları kısıtlı olur)
    # Burada fail etmiyoruz; sadece bilgiyi not düşmek isterseniz log basabilirsiniz.
    _ = _validate_min_schema(out, key_col)

    return out

def load_events(path: str, key_col: str = "geoid") -> pd.DataFrame:
    """
    Olay verisini CSV'den okur, esnek zaman & lat/lon normalize eder.
    DÖNEN DF: her zaman 'ts' (UTC), mümkünse 'latitude'/'longitude' ve/veya 'geoid' içerir.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[load_events] CSV okunamadı: {e}")
        return pd.DataFrame()

    df = _normalize_latlon(df)
    df = normalize_events_ts(df, key_col=key_col)
    return df
