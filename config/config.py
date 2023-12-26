from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()

# Data
DATA_DIR = Path(BASE_DIR, "data")

RAW_DATA_DIR = Path(DATA_DIR, "raw")
RAW_IMG_DIR = Path(RAW_DATA_DIR, "images")
RAW_MASK_DIR = Path(RAW_DATA_DIR, "annotations", "trimaps")
NEW_DATA_DIR = Path(RAW_DATA_DIR, "new_data")

PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")

TRAIN_SET_PATH = Path(PROCESSED_DATA_DIR, "train_set")
VALID_SET_PATH = Path(PROCESSED_DATA_DIR, "valid_set")
TEST_SET_PATH = Path(PROCESSED_DATA_DIR, "test_set")
DEBUGGING_SET_PATH = Path(PROCESSED_DATA_DIR, "debugging_set")

# Config
CONFIG_DIR = Path(BASE_DIR, "config")

# Reports
REPORTS_DIR = Path(BASE_DIR, "reports")
FIGURE_DIR = Path(REPORTS_DIR, "figures")
