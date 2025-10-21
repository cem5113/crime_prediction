# components/utils/loaders.py
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from .constants import (
    # Yol sabitleri
    ARTIFACT_ZIP, ARTIFACTS_DIR,
    DATA_DIR, GRID_FILE,
    # Artefakt içi dosya adları (zip içindeki görünen isimler)
    # Not: Parquet kullanıyoruz
    ART_SF_CSV as ART_SF_PARQUET,   # geriye uyumluluk için isim aynı kaldı
    ART_FR_CSV as ART_FR_PARQUET,   # geriye uyumluluk için isim aynı kaldı
    ART_GRID,
    # Canonical çıktı yolları
    FACT_INCIDENTS, FACT_CELL_TIMESLICES, PRED_CELL_TIMESLICES,
    # Alan sabitleri
    KEY_COL,
    # Kategori normalize yardımcıları (varsa)
    # optional: canonical_category, category_key_list
)

# Artık .parquet: zip içi dosya isimlerini .parquet olarak bekleyeceğiz.
# Eğer constants.py içinde hala "sf_crime_09.csv" yazıyorsa, burada .parquet'a çeviriyoruz.
def _ensure_parquet_name(name: str) -> str:
    name = str(name)
    if name.lower().endswith(".csv"):
        return name[:-4] + ".parquet"
    return name

ART_SF_PARQUET = _ensure_parquet_name(ART_SF_PARQUET)  # "sf_crime_09.parquet"
ART_FR_PARQUET = _ensure_parquet_name(ART_FR_PARQUET)  # "fr_crime_09.parquet"
ART_PRED_PARQUET = "pred_cell_timeslices.parquet"      # opsiyonel model çıktısı


# ---------------------------
# Helpers
# ---------------------------

def _to_dt(s, utc: bool = True) -> pd.Series:
    """Robust datetime parse."""
    return pd.to_datetime(s, errors="coerce", utc=utc)

def _read_parquet_from_zip(zf: zipfile.ZipFile, inner_path: str) -> pd.DataFrame:
    """Zip içinden parquet okuyucu (memory buffer ile)."""
    with zf.open(inner_path) as f:
        buf = io.BytesIO(f.read())
    return pd.read_parquet(buf)

def _write_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _write_bytes(content: bytes, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

def _normalize_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if KEY_COL in df.columns:
        df[KEY_COL] = df[KEY_COL].astype(str)
    return df

def _find_ts_col(df: pd.DataFrame, candidates: List[str] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["ts", "occurred_at", "reported_at", "timestamp", "datetime", "date_time"]
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


# ---------------------------
# Public API
# ---------------------------

@dataclass
class ArtifactIngestResult:
    sf: pd.DataFrame
    fr: pd.DataFrame
    pred_opt: Optional[pd.DataFrame]  # varsa artefakt içinden gelen tahmin
    grid_updated: bool


def import_latest_artifact(save_raw: bool = False) -> ArtifactIngestResult:
    """
    crime_prediction_data/artifacts/latest.zip:
      - sf_crime_09.parquet  (GEOID bazlı zenginleştirilmiş)
      - fr_crime_09.parquet  (olay-ID bazlı zenginleştirilmiş)
      - pred_cell_timeslices.parquet  (opsiyonel; model çıktısı)
      - sf_cells.geojson  (opsiyonel; grid)
    """
    if not ARTIFACT_ZIP.exists():
        raise FileNotFoundError(f"Artefakt bulunamadı: {ARTIFACT_ZIP}")

    with zipfile.ZipFile(ARTIFACT_ZIP, "r") as zf:
        # zorunlu dosyalar
        sf = _read_parquet_from_zip(zf, ART_SF_PARQUET)
        fr = _read_parquet_from_zip(zf, ART_FR_PARQUET)

        # opsiyonel: tahmin parquet
        try:
            pred_opt = _read_parquet_from_zip(zf, ART_PRED_PARQUET)
        except KeyError:
            pred_opt = None

        # opsiyonel: grid
        grid_updated = False
        try:
            with zf.open(ART_GRID) as g:
                geojson_bytes = g.read()
            _write_bytes(geojson_bytes, GRID_FILE)
            grid_updated = True
        except KeyError:
            pass

    # normalize temel kolonlar
    sf = _normalize_sf(sf)
    fr = _normalize_fr(fr)
    if pred_opt is not None:
        pred_opt = _normalize_pred(pred_opt)

    # opsiyonel ham kaydetme (debug amaçlı)
    if save_raw:
        _write_parquet(sf, DATA_DIR / "sf_crime_09.parquet")
        _write_parquet(fr, DATA_DIR / "fr_crime_09.parquet")
        if pred_opt is not None:
            _write_parquet(pred_opt, DATA_DIR / "pred_cell_timeslices_artifact.parquet")

    return ArtifactIngestResult(sf=sf, fr=fr, pred_opt=pred_opt, grid_updated=grid_updated)


def materialize_canonical(sf: pd.DataFrame, fr: pd.DataFrame, pred_opt: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    - fr -> FACT_INCIDENTS
    - sf -> FACT_CELL_TIMESLICES (geoid×hour y_label)
    - pred_opt varsa doğrudan PRED_CELL_TIMESLICES'e yazılır; yoksa baseline üretir.
    """
    # 1) fact_incidents
    fact_inc = _materialize_fact_incidents(fr)
    _write_parquet(fact_inc, FACT_INCIDENTS)

    # 2) fact_cell_timeslices (saatlik)
    fact_cell = _materialize_fact_cell_timeslices(sf)
    _write_parquet(fact_cell, FACT_CELL_TIMESLICES)

    # 3) pred_cell_timeslices
    if pred_opt is not None and not pred_opt.empty:
        pred = _project_pred_columns(pred_opt)
    else:
        pred = _baseline_from_history(fact_cell)

    _write_parquet(pred, PRED_CELL_TIMESLICES)

    return {
        "fact_incidents": str(FACT_INCIDENTS),
        "fact_cell_timeslices": str(FACT_CELL_TIMESLICES),
        "pred_cell_timeslices": str(PRED_CELL_TIMESLICES),
    }


# ---------------------------
# Normalizers / Materializers
# ---------------------------

def _normalize_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    GEOID bazlı özet veri:
      - KEY_COL (geoid) → str
      - 'ts' yoksa day/hour veya date/hour'tan türet
      - 'y_label' yoksa 0 ile doldur
    """
    df = df.copy()
    df = _normalize_geoid(df)

    lower = {str(c).strip().lower(): c for c in df.columns}

    # ts
    ts_col = _find_ts_col(df, candidates=["ts", "date_time", "datetime", "timestamp", "date"])
    if ts_col:
        df["ts"] = _to_dt(df[ts_col])
    else:
        # day/hour veya date/hour varsa saat başına yuvarla
        date_col = lower.get("date") or lower.get("day") or lower.get("date_time")
        hour_col = lower.get("hour") or lower.get("hr")
        if date_col and hour_col:
            df["ts"] = _to_dt(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")
        else:
            df["ts"] = pd.NaT

    # y_label
    if "y_label" not in df.columns:
        # sayım kolonları varsa toplayalım, yoksa 0
        cand = None
        for c in ["count", "n", "num", "label", "y"]:
            if c in lower:
                cand = lower[c]; break
        df["y_label"] = (df[cand] if cand else 0).astype("Int64").fillna(0).astype(int)

    return df


def _normalize_fr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Olay-ID bazlı veri:
      - incident_id
      - KEY_COL (geoid) → str
      - ts kolonunu yakala
      - offense_category/offense_display normalize (mümkünse)
    """
    df = df.copy()
    # incident id
    if "incident_id" not in df.columns:
        for cand in ["id", "event_id", "crime_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "incident_id"})
                break

    df = _normalize_geoid(df)

    ts_col = _find_ts_col(df)
    df["ts"] = _to_dt(df[ts_col]) if ts_col else pd.NaT

    # kategori isimleri: offense_category varsa taşır, yoksa type/offense/category'den türetir
    if "offense_category" not in df.columns:
        for cand in ["type", "offense", "category", "offense_type"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "offense_category"})
                break

    # UI'da kullanılmak üzere offense_display (Title Case) oluşturmaya çalış
    try:
        from .constants import canonical_category
        df["offense_display"] = df["offense_category"].apply(canonical_category) if "offense_category" in df.columns else None
    except Exception:
        pass

    return df


def _normalize_pred(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pred tablo normalize: beklenen ana kolonlar:
      - timeslice_start (UTC), timeslice_end (UTC)
      - geoid (KEY_COL), pred_expected (λ), risk_score (0-1)
    Ek kolonlar bozulmadan taşınır.
    """
    df = df.copy()
    df = _normalize_geoid(df)

    # zaman kolonları
    if "timeslice_start" in df.columns:
        df["timeslice_start"] = _to_dt(df["timeslice_start"])
    elif "ts" in df.columns:
        df = df.rename(columns={"ts": "timeslice_start"})
        df["timeslice_start"] = _to_dt(df["timeslice_start"])
    else:
        # ts yoksa yapacak bir şey yok; boş bırakalım
        df["timeslice_start"] = pd.NaT

    if "timeslice_end" in df.columns:
        df["timeslice_end"] = _to_dt(df["timeslice_end"])
    else:
        df["timeslice_end"] = df["timeslice_start"] + pd.Timedelta(hours=1)

    # pred_expected
    if "pred_expected" not in df.columns:
        # lambda / expected gibi isimleri yakala
        for cand in ["lambda", "expected", "e", "mean"]:
            if cand in df.columns:
                df["pred_expected"] = pd.to_numeric(df[cand], errors="coerce")
                break
        else:
            df["pred_expected"] = 0.0

    # risk_score ~ P(>=1) ≈ 1 - exp(-λ)
    if "risk_score" not in df.columns:
        lam = pd.to_numeric(df["pred_expected"], errors="coerce").fillna(0.0)
        df["risk_score"] = (1 - np.exp(-lam)).clip(0, 1)

    return df


def _materialize_fact_incidents(fr: pd.DataFrame) -> pd.DataFrame:
    keep = ["incident_id", KEY_COL, "ts"]
    if "offense_category" in fr.columns:
        keep.append("offense_category")
    if "offense_display" in fr.columns:
        keep.append("offense_display")
    # diğer zengin kolonları da koru
    keep += [c for c in fr.columns if c not in set(keep)]
    fact_inc = fr[keep].copy()
    return fact_inc


def _materialize_fact_cell_timeslices(sf: pd.DataFrame) -> pd.DataFrame:
    df = sf.copy()
    df["ts"] = _to_dt(df["ts"]).dt.floor("h")
    df = df.dropna(subset=["ts", KEY_COL])

    # saatlik geoid × ts sayımı (y_label toplanır)
    fact_cell = (
        df.groupby([KEY_COL, "ts"], as_index=False)
          .agg(y_label=("y_label", "sum"))
    )
    return fact_cell


def _project_pred_columns(pred: pd.DataFrame) -> pd.DataFrame:
    """
    pred dataframe'ini PRED_CELL_TIMESLICES şemasına indirger.
    """
    keep = ["timeslice_start", "timeslice_end", KEY_COL, "risk_score", "pred_expected"]
    # eksik olanları doldur
    if "timeslice_start" not in pred.columns and "ts" in pred.columns:
        pred = pred.rename(columns={"ts": "timeslice_start"})
    if "timeslice_end" not in pred.columns:
        pred["timeslice_end"] = _to_dt(pred["timeslice_start"]) + pd.Timedelta(hours=1)
    if "risk_score" not in pred.columns and "pred_expected" in pred.columns:
        lam = pd.to_numeric(pred["pred_expected"], errors="coerce").fillna(0.0)
        pred["risk_score"] = (1 - np.exp(-lam)).clip(0, 1)
    # tip güvenliği
    pred["timeslice_start"] = _to_dt(pred["timeslice_start"])
    pred["timeslice_end"]   = _to_dt(pred["timeslice_end"])
    pred = _normalize_geoid(pred)
    return pred[[k for k in keep if k in pred.columns]].copy()


def _baseline_from_history(cell_df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Basit baseline tahmin:
      - son window_days gün için geo × (weekday,hour) ortalama
      - 24 saatlik ileri ufuk (her geoid için)
    """
    df = cell_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["timeslice_start", "timeslice_end", KEY_COL, "risk_score", "pred_expected"])

    df["weekday"] = df["ts"].dt.weekday
    df["hour"]    = df["ts"].dt.hour

    cutoff = df["ts"].max() - pd.Timedelta(days=window_days)
    hist = df[df["ts"] >= cutoff].copy()

    lam_tbl = (
        hist.groupby([KEY_COL, "weekday", "hour"], as_index=False)["y_label"]
            .mean().rename(columns={"y_label": "lambda"})
    )

    ref_time = df["ts"].max().ceil("h")
    horizon  = 24
    future   = pd.DataFrame({"timeslice_start": pd.date_range(ref_time, periods=horizon, freq="h", tz="UTC")})
    future["weekday"] = future["timeslice_start"].dt.weekday
    future["hour"]    = future["timeslice_start"].dt.hour

    geoids = df[KEY_COL].dropna().unique()
    rows = []
    for gid in geoids:
        tmp = future.merge(lam_tbl[lam_tbl[KEY_COL] == gid], on=["weekday", "hour"], how="left")
        tmp[KEY_COL] = gid
        tmp["lambda"] = pd.to_numeric(tmp["lambda"], errors="coerce").fillna(0.05)
        rows.append(tmp[[KEY_COL, "timeslice_start", "lambda"]])

    pred = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[KEY_COL, "timeslice_start", "lambda"])
    pred["timeslice_end"] = pred["timeslice_start"] + pd.Timedelta(hours=1)
    pred["pred_expected"] = pred["lambda"].astype(float).clip(lower=0)
    pred["risk_score"]    = (1 - np.exp(-pred["pred_expected"])).clip(0, 1)

    return pred[["timeslice_start", "timeslice_end", KEY_COL, "risk_score", "pred_expected"]]
