# utils/constants.py
SF_TZ_OFFSET = -7
CRIME_TYPES = ["assault", "burglary", "theft", "robbery", "vandalism"]
KEY_COL = "geoid"
CACHE_VERSION = "v2-geo-poisson"
MODEL_VERSION = "v0.1.0"
MODEL_LAST_TRAIN = "2025-10-04"

CATEGORIES = ["Assault","Burglary","Robbery","Theft","Vandalism","Vehicle Theft"]
DAYS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
SEASONS = ["Winter","Spring","Summer","Autumn"]

CATEGORY_TO_KEYS = {
    "Assault": ["assault"],
    "Burglary": ["burglary"],
    "Robbery": ["robbery"],
    "Theft": ["theft", "larceny"],
    "Vandalism": ["vandalism"],
    "Vehicle Theft": ["vehicle_theft", "auto_theft", "motor_vehicle_theft"],
}
