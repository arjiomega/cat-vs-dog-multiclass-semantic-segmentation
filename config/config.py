from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()

# Data
DATA_DIR = Path(BASE_DIR,"data")

RAW_DATA_DIR = Path(DATA_DIR,"raw")
RAW_IMG_DIR = Path(RAW_DATA_DIR,"images")
RAW_MASK_DIR = Path(RAW_DATA_DIR,"annotations","trimaps")

PROCESSED_DATA_DIR = Path(DATA_DIR,"processed")
PROCESSED_IMG_DIR = Path(PROCESSED_DATA_DIR,"img")
PROCESSED_MASK_DIR = Path(PROCESSED_DATA_DIR,"mask")

# Generate Directories
PROCESSED_IMG_DIR.mkdir(parents=True,exist_ok=True)
PROCESSED_MASK_DIR.mkdir(parents=True,exist_ok=True)

# Config
CONFIG_DIR = Path(BASE_DIR,"config")

# Reports
REPORTS_DIR = Path(BASE_DIR,"reports")
FIGURE_DIR = Path(REPORTS_DIR,"figures")