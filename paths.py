import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = ROOT_DIR / "data"
GLOVE_DIR = DATA_DIR / "glove.6B"
