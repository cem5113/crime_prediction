## 1) src/config.py

```python
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class Paths:
    CRIME_DIR: str = os.getenv("CRIME_DATA_DIR", "crime_data")
    GEO_DIR: str = os.path.join(CRIME_DIR, "geo")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")

    # Inputs (opsiyonel)
    SF_CRIME_CSV: str = os.path.join(CRIME_DIR, "sf_crime.csv")
    SF50_CSV: str = os.path.join(CRIME_DIR, "sf_crime_50.csv")
    SF52_CSV: str = os.path.join(CRIME_DIR, "sf_crime_52.csv")
    TRACTS_GEOJSON: str = os.path.join(GEO_DIR, "tracts.geojson")
    TRACT_CENTROIDS_CSV: str = os.path.join(GEO_DIR, "tract_centroids.csv")

    # Models (opsiyonel)
    STACK_MODEL: str = os.path.join(MODEL_DIR, "stacking_model.joblib")
    CALIBRATOR: str = os.path.join(MODEL_DIR, "calibrator.joblib")
    CONFORMAL_RESIDUALS: str = os.path.join(MODEL_DIR, "calibration_residuals.npy")

    # Outputs
    RISK_DIR: str = os.path.join(CRIME_DIR, "risk")

@dataclass
class Params:
    GEOID_LEN: int = int(os.getenv("GEOID_LEN", "11"))
    TOP_K: int = int(os.getenv("PATROL_TOP_K", "10"))
    ALERT_THR: float = float(os.getenv("ALERT_THR", "0.35"))
    HOUR_BIN: int = 2  # saat aralığı genişliği (saat)

    # rota
    N_TEAMS: int = int(os.getenv("PATROL_TEAMS", "2"))
    TIME_BUDGET_MIN: int = int(os.getenv("PATROL_MIN", "60"))

paths = Paths()
params = Params()
