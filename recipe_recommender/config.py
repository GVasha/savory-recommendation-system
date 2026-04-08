from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RECIPES_PATH = DATA_DIR / "recipes_visible_only.csv"
DEFAULT_INTERACTIONS_PATH = DATA_DIR / "interactions_train.csv"

# Core model defaults
DEFAULT_TOP_K = 10
DEFAULT_RANDOM_STATE = 42

# Hybrid weights for item-item similarity
DEFAULT_W_TEXT = 0.8
DEFAULT_W_NUMERIC = 0.15
DEFAULT_W_CATEGORY = 0.05
