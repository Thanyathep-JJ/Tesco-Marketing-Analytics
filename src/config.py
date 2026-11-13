"""Project paths and constants."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Raw parquet files live outside the repo (licensed data, not committed).
# Override with the TESCO_DATA_DIR environment variable if stored elsewhere.
DATA_DIR = Path(os.getenv("TESCO_DATA_DIR", REPO_ROOT.parent / "data" / ".parquet"))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIG_DIR = REPO_ROOT / "reports" / "figures"

MAX_DAY = 711          # last day in transaction_data
SNAPSHOT_DAY = MAX_DAY + 1
N_WEEKS = 102
RANDOM_STATE = 42

# Segment display order and colours (shared by all notebooks)
SEGMENT_ORDER = ["Champions", "Loyal", "At Risk", "Occasional"]
SEGMENT_TH = {
    "Champions": "ประจำ / มูลค่าสูง",
    "Loyal": "ซื้อซ้ำ / มูลค่าปานกลาง",
    "At Risk": "เริ่มห่างหาย",
    "Occasional": "ซื้อเป็นครั้งคราว",
}
SEGMENT_COLORS = {
    "Champions": "#2a9d8f",
    "Loyal": "#7ba7e0",
    "At Risk": "#e76f51",
    "Occasional": "#9e9e9e",
}

for _d in (PROCESSED_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
