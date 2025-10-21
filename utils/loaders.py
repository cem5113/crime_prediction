# components/utils/loaders.py
import io
import json
import zipfile
import pandas as pd
import numpy as np
from .constants import (
    ARTIFACT_ZIP, ART_SF_CSV, ART_FR_CSV, ART_GRID,
    DATA_DIR, GRID_FILE,
    FACT_INCIDENTS, FACT_CELL_TIMESLICES, PRED_CELL_TIMESLICES,
    KEY_COL,
)

def _to_dt(s, utc=True):
    return pd.to_datetime(s, errors="coerce", utc=utc)

def import_latest_artifact(save_raw: bool = False) -> dict:
    """
    latest.zip içinden sf_crime_09.csv ve fr_crime_09.csv'yi okur,
    gerekirse grid'i günceller; dict döner.
    """
    if not ARTIFACT_ZIP.exists():
        raise FileNotFoundError(f"Artefakt bulunamadı: {ARTIFACT_ZIP}")
    out = {}
    with zipfile.ZipFile(ARTIFACT_ZIP, "r") as zf:
        # GEOID tabanlı
        with zf.open(ART_SF_CSV) as f:
            sf = pd.read_csv(f)
        # Olay-ID tabanlı
        with zf.open(ART_FR_CSV) as f:
            fr = pd.read_csv(f)

        # Grid varsa güncelle (opsiyonel)
        try:
            with zf.open(ART_GRID) as g:
                geojson_bytes = g.read()
            GRID_FILE.write_bytes(geojson_bytes)
        except KeyError:
            pass  # artefakta grid olmayabilir → components/data’daki kullanılır

    # normalize
    sf = _normalize_sf(sf)
    fr = _normalize_fr(fr)

    if save_raw:
        (DATA_DIR / "sf_crime_09.csv").write_text(sf.to_csv(index=False), encoding="utf-8")
        (DATA_DIR / "fr_crime_09.csv").write_text(fr.to_csv(index=False), encoding="utf-8")

    out["sf"], out["fr"] = sf, fr
    return out

def _normalize_sf(df: pd.DataFrame) -> pd.DataFrame:
    # GEOID string
    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str)
    # zaman: varsa ts; yoksa date/hour → ts
    lower = {c.lower(): c for c in df.columns}
    if "ts" in lower:
        df["ts"] = _to_dt(df[lower["ts"]])
    else:
        date_col = lower.get("date") or lower.get("day") or lower.get("date_time")
        hour_col = lower.get("hour") or lower.get("hr")
        if date_col and hour_col:
            df["ts"] = _to_dt(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")
        else:
            df["ts"] = pd.NaT
    # y / count alanı: yoksa 0
    if "y_label" not in df.columns:
        if "count" in lower:
            df["y_label"] = df[lower["count"]].clip(lower=0).astype(int)
        else:
            df["y_label"] = 0
    return df

def _normalize_fr(df: pd.DataFrame) -> pd.DataFrame:
    # incident id
    if "incident_id" not in df.columns:
        # olası alternatif ad
        for cand in ["id", "event_id", "crime_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "incident_id"}); break
    # geoid
    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str)
    # ts/occurred_at/reported_at
    ts_col = None
    for c in ["ts","occurred_at","reported_at","timestamp","datetime","date_time"]:
        if c in df.columns: ts_col = c; break
    df["ts"] = _to_dt(df[ts_col]) if ts_col else pd.NaT
    # kategori standardizasyonu
    if "offense_category" not in df.columns:
        for c in ["type","offense","category"]:
            if c in df.columns:
                df = df.rename(columns={c: "offense_category"}); break
    return df

# ----------------- Canonical veri setleri -----------------

def materialize_canonical(sf: pd.DataFrame, fr: pd.DataFrame) -> dict:
    """
    sf -> fact_cell_timeslices, fr -> fact_incidents
    Ayrıca basit bir baseline ile pred_cell_timeslices üretir (λ ~ son 7 gün/h saat/geo).
    """
    # fact_incidents
    fact_inc = fr.copy()
    keep_inc = ["incident_id","geoid","ts","offense_category"]
    keep_inc += [c for c in fr.columns if c not in keep_inc]  # zengin kolonları koru
    fact_inc = fact_inc[keep_inc]
    fact_inc.to_parquet(FACT_INCIDENTS, index=False)

    # fact_cell_timeslices: saatlik geoid × ts sayımı
    cell = (
        sf.dropna(subset=["geoid"])
          .assign(ts=lambda d: _to_dt(d["ts"]).dt.floor("h"))
          .dropna(subset=["ts"])
          .groupby([KEY_COL, "ts"], as_index=False)
          .agg(y_label=("y_label","sum"))
    )
    cell.to_parquet(FACT_CELL_TIMESLICES, index=False)

    # Basit baseline tahmin (λ): son 7 günün saatlik ortalaması
    pred = _baseline_from_history(cell)
    pred.to_parquet(PRED_CELL_TIMESLICES, index=False)

    return {
        "fact_incidents": FACT_INCIDENTS,
        "fact_cell_timeslices": FACT_CELL_TIMESLICES,
        "pred_cell_timeslices": PRED_CELL_TIMESLICES,
    }

def _baseline_from_history(cell_df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Saat-of-day + weekday etkisiyle basit λ tahmini."""
    df = cell_df.copy()
    df["weekday"] = df["ts"].dt.weekday
    df["hour"]    = df["ts"].dt.hour
    # son window_days: (eğer tüm tarih yoksa hepsini kullanır)
    cutoff = df["ts"].max() - pd.Timedelta(days=window_days) if not df.empty else None
    if cutoff is not None:
        hist = df[df["ts"] >= cutoff].copy()
    else:
        hist = df

    # geo × (weekday,hour) ortalama
    lam = (
        hist.groupby([KEY_COL,"weekday","hour"], as_index=False)["y_label"]
            .mean().rename(columns={"y_label":"lambda"})
    )

    # En güncel saat dilimi için tahmin tablosu
    ref_time = df["ts"].max().ceil("h") if not df.empty else pd.Timestamp.utcnow().ceil("h")
    horizon  = 24  # Home/Forecast 0–24h için 24 saatlik grid
    future   = pd.DataFrame({"ts": pd.date_range(ref_time, periods=horizon, freq="h", tz="UTC")})
    future["weekday"] = future["ts"].dt.weekday
    future["hour"]    = future["ts"].dt.hour

    # tüm geoidler
    geoids = df[KEY_COL].dropna().unique()
    pred_rows = []
    for gid in geoids:
        tmp = future.merge(lam[lam[KEY_COL]==gid], on=["weekday","hour"], how="left")
        tmp[KEY_COL] = gid
        tmp["lambda"] = tmp["lambda"].fillna(0.05)  # çok az veri varsa küçük taban
        pred_rows.append(tmp[[KEY_COL,"ts","lambda"]])

    pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(columns=[KEY_COL,"ts","lambda"])
    pred = pred.rename(columns={"ts":"timeslice_start"})
    pred["timeslice_end"] = pred["timeslice_start"] + pd.Timedelta(hours=1)
    pred["pred_expected"] = pred["lambda"].astype(float).clip(lower=0)
    pred["risk_score"]    = (1 - np.exp(-pred["pred_expected"])).clip(0,1)  # P(≥1) ~ 1 - e^-λ
    return pred[[ "timeslice_start","timeslice_end", KEY_COL, "risk_score","pred_expected"]]
