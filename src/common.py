## 2) src/common.py

from __future__ import annotations
import os
import re
import pandas as pd
from typing import Optional

HOUR_LABELS = [(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,22),(22,24)]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clean_geoid(s: pd.Series, width: int = 11) -> pd.Series:
    out = s.astype(str).str.extract(r"(\d+)")[0].str[:width].str.zfill(width)
    return out

def to_hour_range(hour: int, width: int = 2) -> str:
    h0 = (hour // width) * width
    h1 = min(h0 + width, 24)
    return f"{h0:02d}:00–{h1:02d}:00"

def infer_hour_from_col(df: pd.DataFrame) -> pd.Series:
    for c in ["hour", "event_hour", "HOUR", "hour_of_day"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).clip(0,23)
    # fallback
    return pd.Series([0]*len(df), index=df.index, dtype=int)

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.isfile(path):
            return pd.read_csv(path, low_memory=False)
    except Exception:
        pass
    return None

def minmax01(x):
    import numpy as np
    if len(x)==0: return x
    a,b = float(min(x)), float(max(x))
    if b-a < 1e-9:
        return [0.0 for _ in x]
    return [(xi-a)/(b-a) for xi in x]
