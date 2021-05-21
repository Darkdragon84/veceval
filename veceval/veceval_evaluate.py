import os

from veceval.settings import AvailableTasks, CONFIGS_FOLDER, MODES
import importlib


def compile_trainers():
    task_to_data_training = {}
    for task in AvailableTasks:
        for mode in MODES:
            base_name = f"{task.value}_{mode}"
            config_file_path = CONFIGS_FOLDER / f"config_{base_name}.txt"
            if not os.path.isfile(config_file_path):
                continue
            try:
                trainer = importlib.import_module(f"veceval.training.{base_name}")
            except (ImportError, ModuleNotFoundError):
                continue
    return task_to_data_training


TASK_TO_DATA_TRAINING = {
    task.name: {
        "config": CONFIGS_FOLDER / f"config_{task.value}_{mode}.txt",
        "trainer": importlib.import_module(f"training")}
    for task in AvailableTasks
    for mode in MODES
}

def main():
    pass

if __name__ == '__main__':
    main()
