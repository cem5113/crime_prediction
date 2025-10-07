# utils/reports.py
from __future__ import annotations
import pandas as pd
import numpy as np

_TS_CANDIDATES = [
    "ts","timestamp","datetime","date_time","reported_at","occurred_at","time","date"
]

def _parse_ts_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        med = pd.to_numeric(s, errors="coerce").dropna().astype(float).median()
        unit = "ms" if (pd.notna(med) and med > 1e12) else "s"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s.astype(str).str.strip(), utc=True, errors="coerce")

def normalize_events_ts(df: pd.DataFrame | None, key_col: str = "geoid") -> pd.DataFrame:
    """
    Olay verisinde esnek zaman sütunu normalizasyonu.
    Çıktı: ts(utc), hour, dow, date (+ varsa key_col)
    Boş veya parse edilemeyen durumda boş DataFrame döner.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    lc = {c.lower(): c for c in df.columns}
    ts = None
    # 1) doğrudan adaylar
    for name in _TS_CANDIDATES:
        if name in lc:
            ts = _parse_ts_series(df[lc[name]])
            break
    # 2) date + time birleşimi
    if ts is None and ("date" in lc and "time" in lc):
        tmp = df[[lc["date"], lc["time"]]].astype(str).agg(" ".join, axis=1)
        ts = pd.to_datetime(tmp, utc=True, errors="coerce")

    if ts is None:
        return pd.DataFrame()

    out = df.copy()
    out["ts"] = ts
    out = out.dropna(subset=["ts"])
    if out.empty:
        return pd.DataFrame()

    out["hour"] = out["ts"].dt.hour
    out["dow"]  = out["ts"].dt.dayofweek
    out["date"] = out["ts"].dt.date

    # key_col adı farklı büyük/küçük olabilir → varsa bırak
    if key_col not in out.columns and key_col.lower() in lc:
        out.rename(columns={lc[key_col.lower()]: key_col}, inplace=True)

    return out

def load_events(path: str) -> pd.DataFrame:
    """
    Basit CSV yükleyici. ts veya timestamp (vs) + (geoid ya da lat/lon) bekler.
    Eksikse yine de df döndürür (boş da olabilir).
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df
