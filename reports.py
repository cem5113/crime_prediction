# utils/reports.py
import pandas as pd

def load_events(path: str) -> pd.DataFrame:
    """
    Olay verilerini CSV'den yükler.
    CSV'de en az şu kolonlar olmalı: ts, lat, lon
    """
    try:
        df = pd.read_csv(path, parse_dates=["ts"])
    except Exception as e:
        print(f"[load_events] CSV okunamadı: {e}")
        return pd.DataFrame()

    # Eksik kolonları kontrol et
    for col in ["ts", "lat", "lon"]:
        if col not in df.columns:
            print(f"[load_events] Eksik kolon: {col}")
            return pd.DataFrame()

    return df
