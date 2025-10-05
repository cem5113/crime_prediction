# utils/reports.py
import pandas as pd

def load_events(path: str) -> pd.DataFrame:
    """
    Olay verilerini CSV'den yükler.
    CSV'de en az şu kolonlar olmalı: ts, lat, lon (veya geoid)
    """
    try:
        df = pd.read_csv(path, parse_dates=["ts"])
    except Exception as e:
        print(f"[load_events] CSV okunamadı: {e}")
        return pd.DataFrame()

    needed_any = ({"geoid"} <= set(df.columns)) or ({"lat", "lon"} <= set(df.columns))
    if "ts" not in df.columns or not needed_any:
        print("[load_events] Eksik kolonlar. Gerekli: ts ve (geoid) ya da (lat, lon).")
        return pd.DataFrame()

    return df
