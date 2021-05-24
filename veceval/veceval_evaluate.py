import os
import sys
from functools import partial

from veceval.settings import AvailableTasks, CONFIGS_FOLDER, MODES
import importlib


def compile_trainers():
    task_to_trainer = {}
    for task in AvailableTasks:
        for mode in MODES:
            base_name = f"{task.value}_{mode}"
            config_file_path = CONFIGS_FOLDER / f"config_{base_name}.txt"
            if not os.path.isfile(config_file_path):
                continue
            try:
                trainer = getattr(importlib.import_module(f"veceval.training.{base_name}"), "main_training")
                trainer = partial(trainer, config_file_path)
            except (ImportError, ModuleNotFoundError, AttributeError):
                continue
            task_to_trainer[base_name] = trainer
    return task_to_trainer


def main(embedding_name):
    task_to_trainer = compile_trainers()
    for task_name, trainer in task_to_trainer.items():
        print(80 * "=")
        print(task_name.upper())
        print(80 * "-")
        trainer(embedding_name)
        print()


if __name__ == '__main__':
    main(sys.argv[1])
