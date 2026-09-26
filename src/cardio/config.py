"""Project-wide constants: paths, column groups and data plausibility bounds."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "HeartFailureDataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

ID_COL = "id"
TARGET_COL = "cardio"

# Raw feature columns as they appear in the CSV
RAW_FEATURES = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

# Column groups after feature engineering (see features.py)
CONTINUOUS_FEATURES = [
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "bmi",
    "pulse_pressure",
    "map",
]
ORDINAL_FEATURES = ["cholesterol", "gluc"]
BINARY_FEATURES = ["gender", "smoke", "alco", "active"]

# Inclusive plausibility bounds used to drop data-entry errors
PLAUSIBLE_RANGES = {
    "ap_hi": (80, 250),
    "ap_lo": (40, 160),
    "height": (120, 220),
    "weight": (30, 200),
}
