import os

from pathlib import Path


def valid_path(path):
    return os.path.isdir(path.as_posix()) and "_" not in path.stem


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_FOLDER = PROJECT_ROOT / Path("training") / Path("configs")
DATA_FOLDER = PROJECT_ROOT / Path("data")

AVAILABLE_MODES = ["fixed", "finetuned"]
AVAILABLE_TASK_TO_DATASET = {
    task: [
        dataset for dataset in os.listdir(DATA_FOLDER / task) if valid_path(DATA_FOLDER / task / dataset)
    ]
    for task in os.listdir(DATA_FOLDER) if valid_path(DATA_FOLDER / task)
}

MODES = AVAILABLE_MODES
TASK_TO_DATASET = AVAILABLE_TASK_TO_DATASET

AFFILIATION = "iris.ai"


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

MODES = [mode for mode in MODES if mode in AVAILABLE_MODES]
TASK_TO_DATASET = {
    task: [dataset for dataset in datasets if dataset in AVAILABLE_TASK_TO_DATASET[task]]
    for task, datasets in TASK_TO_DATASET.items() if task in AVAILABLE_TASK_TO_DATASET
}

os.makedirs(EXT_DATA_ROOT, exist_ok=True)
os.makedirs(RUNNING_LOG_FILE_PATH.parent, exist_ok=True)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
os.makedirs(PICKLES_FOLDER, exist_ok=True)
