# components/utils/constants.py
from __future__ import annotations
from pathlib import Path

# ─────────────────────────────
# Proje yolları / artefaktlar
# ─────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
COMP_DIR    = PROJECT_DIR / "components"
DATA_DIR    = COMP_DIR / "data"

ARTIFACTS_DIR = PROJECT_DIR / "crime_prediction_data" / "artifacts"
ARTIFACT_ZIP  = ARTIFACTS_DIR / "latest.zip"

# Artefakt içindeki beklenen dosya adları (zip içi isimler)
ART_SF_CSV = "sf_crime_09.csv"   # GEOID bazlı zenginleştirilmiş veri
ART_FR_CSV = "fr_crime_09.csv"   # olay-ID bazlı zenginleştirilmiş veri
ART_GRID   = "sf_cells.geojson"  # (opsiyonel) grid dosyası

# Canonical/çıktı dosyaları (UI bunlardan beslenecek)
FACT_INCIDENTS       = DATA_DIR / "fact_incidents.parquet"
FACT_CELL_TIMESLICES = DATA_DIR / "fact_cell_timeslices.parquet"
PRED_CELL_TIMESLICES = DATA_DIR / "pred_cell_timeslices.parquet"

# Grid (fallback)
GRID_FILE = DATA_DIR / "sf_cells.geojson"

# ─────────────────────────────
# Saat dilimi / model meta
# ─────────────────────────────
SF_TZ_OFFSET     = -7
KEY_COL          = "geoid"

# Model & cache meta
CACHE_VERSION    = "v2-geo-poisson"
MODEL_VERSION    = "v0.1.0"          # (senin verdiğin)
MODEL_LAST_TRAIN = "2025-10-04"      # (senin verdiğin)

# ─────────────────────────────
# Kategoriler: tek sözleşme
# ─────────────────────────────
# 1) UI'de görünen "başlık" adları (Title Case)
DISPLAY_CATEGORIES = [
    "Assault", "Burglary", "Robbery", "Theft", "Vandalism", "Vehicle Theft"
]

# 2) Model/ham veri tarafında beklenen alt anahtarlar (lowercase)
#    - normalize ederken bu anahtarlarla eşleştiriyoruz.
CATEGORY_TO_KEYS = {
    "Assault":        ["assault"],
    "Burglary":       ["burglary"],
    "Robbery":        ["robbery"],
    "Theft":          ["theft", "larceny"],
    "Vandalism":      ["vandalism"],
    "Vehicle Theft":  ["vehicle_theft", "auto_theft", "motor_vehicle_theft", "vehicle"],
}

# 3) Eski kullanımlar için geriye dönük uyumluluk (deprecated):
#    - CRIME_TYPES: tamamen lowercase kısa liste
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]

#    - CATEGORIES: legacy ad; bazı modüller bunu import ediyor olabilir.
#      İçeriği DISPLAY_CATEGORIES ile aynı olacak şekilde expose ediyoruz.
CATEGORIES = DISPLAY_CATEGORIES

# ─────────────────────────────
# Zaman/etiket sabitleri
# ─────────────────────────────
DAYS     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
SEASONS  = ["Winter","Spring","Summer","Autumn"]

# ─────────────────────────────
# Yardımcılar
# ─────────────────────────────
def canonical_category(raw: str | None) -> str | None:
    """
    Girdi: ham kategori ('auto_theft', 'LARCENY', 'Vehicle', 'assault' vb.)
    Çıktı: UI başlık adına normalize ('Vehicle Theft', 'Theft', 'Assault' ...)

    Eşleşme bulunamazsa None döner.
    """
    if not raw:
        return None
    r = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    for display, keys in CATEGORY_TO_KEYS.items():
        if r in keys:
            return display
        # 'vehicle' gibi kısmi eşleşmeler için basit kapsama:
        if any(r in k or k in r for k in keys):
            return display
    return None

def category_key_list(display_name: str) -> list[str]:
    """UI adından (Title Case) model anahtar listesine (lowercase) geçiş."""
    return CATEGORY_TO_KEYS.get(display_name, [])
