import os

from pathlib import Path

AVAILABLE_TASKS = ["chunk", "ner", "nli", "pos", "questions", "sentiment"]
AVAILABLE_MODES = ["fixed", "finetuned"]
AVAILABLE_DATASETS = {
    "ner": ["conll", "chemdner"]
}

TASKS = AVAILABLE_TASKS
MODES = AVAILABLE_MODES
TASK_TO_DATASET = {task: datasets[0] for task, datasets in AVAILABLE_DATASETS.items()}

AFFILIATION = "iris.ai"

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_FOLDER = PROJECT_ROOT / Path("training") / Path("configs")

# create settings_local.py file and override the desired settings
try:
    from veceval.settings_local import EXT_DATA_ROOT
except ImportError:
    EXT_DATA_ROOT = PROJECT_ROOT / "ext_data"
    print('No settings_local.py, using settings values from settings.py')

RUNNING_LOG_FILE_PATH = EXT_DATA_ROOT / Path("logs/log.tsv")
CHECKPOINT_FOLDER = EXT_DATA_ROOT / Path("checkpoints")
PICKLES_FOLDER = EXT_DATA_ROOT / Path("pickles")

# create settings_local.py file and override the desired settings
try:
    from veceval.settings_local import *
except ImportError:
    print('No settings_local.py, using settings values from settings.py')

TASKS = [task for task in TASKS if task in AVAILABLE_TASKS]
MODES = [mode for mode in MODES if mode in AVAILABLE_MODES]

TASK_TO_DATASET = {
    task: dataset if dataset in AVAILABLE_DATASETS[task] else AVAILABLE_DATASETS[task][0]
    for task, dataset in TASK_TO_DATASET.items() if task in AVAILABLE_TASKS
}

os.makedirs(EXT_DATA_ROOT, exist_ok=True)
os.makedirs(RUNNING_LOG_FILE_PATH.parent, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(PICKLES_FOLDER, exist_ok=True)
