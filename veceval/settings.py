import os

from pathlib import Path
from enum import Enum

FIXED = "fixed"
FINETUNED = "finetuned"


class AvailableTasks(Enum):
    CHUNK = "chunk"
    NER = "ner"
    NLI = "nli"
    POS = "pos"
    QUESTIONS = "questions"
    SENTIMENT = "sentiment"


TASKS = [task.value for task in AvailableTasks]
MODES = [FIXED, FINETUNED]
PROJECT_ROOT = Path(__file__).resolve().parent

EXT_DATA_ROOT = PROJECT_ROOT / "ext_data"
AFFILIATION = "iris.ai"

RUNNING_LOG_FILE = EXT_DATA_ROOT / Path("logs/log.txt")
CHECKPOINT_FOLDER = EXT_DATA_ROOT / Path("checkpoints")
PICKLES_FOLDER = EXT_DATA_ROOT / Path("pickles")
CONFIGS_FOLDER = PROJECT_ROOT / Path("training") / Path("configs")

# create settings_local.py file and override the desired settings
try:
    from veceval.settings_local import *
except ImportError:
    print('No settings_local.py, using settings values from settings.py')

os.makedirs(EXT_DATA_ROOT, exist_ok=True)
os.makedirs(RUNNING_LOG_FILE.parent, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(PICKLES_FOLDER, exist_ok=True)
