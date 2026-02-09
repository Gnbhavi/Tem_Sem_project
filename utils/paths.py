import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "Dataset"
CAN_DIR = DATASET_DIR / "Can_Hold"
CANNOT_HOLD_DIR = DATASET_DIR / "Cannot_Hold"
CHART_DIR = ROOT / "Charts"
# os.makedirs(CHART_DIR, exist_ok=True)
# CURRENT_DATA_TIME = datetime.now().strftime('%Y-%m-%d')