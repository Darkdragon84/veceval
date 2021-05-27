import os
import sys
from functools import partial

from veceval.settings import CONFIGS_FOLDER, MODES, TASKS
import importlib


def compile_trainers():
    task_to_trainer = {}
    for task in TASKS:
        for mode in MODES:
            base_name = f"{task}_{mode}"
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


def evaluate_embedding(embedding_name):
    task_to_trainer = compile_trainers()
    for task_name, trainer in task_to_trainer.items():
        print(80 * "-")
        print(task_name.upper())
        print(80 * "-")
        trainer(embedding_name)
        print()


if __name__ == '__main__':
    for emb_name in sys.argv[1:]:
        print(80 * "=")
        print(emb_name)
        print(80 * "=")
        evaluate_embedding(emb_name)
