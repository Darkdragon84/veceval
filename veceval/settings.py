import os

from pathlib import Path

EXT_DATA_ROOT = Path(__file__).resolve().parent / "ext_data"
AFFILIATION = "iris.ai"

RUNNING_LOG_FILE = EXT_DATA_ROOT / Path("logs/log.txt")
CHECKPOINT_FOLDER = EXT_DATA_ROOT / Path("checkpoints")
PICKLES_FOLDER = EXT_DATA_ROOT / Path("pickles")

# create settings_local.py file and override the desired settings
try:
    from veceval.settings_local import *
except ImportError:
    print('No settings_local.py, using settings values from settings.py')

os.makedirs(EXT_DATA_ROOT, exist_ok=True)
os.makedirs(RUNNING_LOG_FILE.parent, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(PICKLES_FOLDER, exist_ok=True)
