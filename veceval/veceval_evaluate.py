import os
import sys
from functools import partial

from veceval.settings import CONFIGS_FOLDER, MODES, TASK_TO_DATASET
import importlib


def compile_trainers():
    task_dataset_trainers = []
    for task, datasets in TASK_TO_DATASET.items():
        for dataset in datasets:
            for mode in MODES:
                base_name = f"{task}_{mode}"
                config_file_path = CONFIGS_FOLDER / f"config_{base_name}.txt"
                if not os.path.isfile(config_file_path):
                    continue
                try:
                    trainer = getattr(importlib.import_module(f"veceval.training.{base_name}"), "main_training")
                    trainer = partial(trainer, config_file_path, dataset)
                except (ImportError, ModuleNotFoundError, AttributeError):
                    continue
                task_dataset_trainers.append((base_name, dataset, trainer))
    return task_dataset_trainers


def evaluate_embedding(embedding_name):
    task_dataset_trainers = compile_trainers()
    for task_name, dataset, trainer in task_dataset_trainers:
        print(80 * "-")
        print(f"{task_name.upper()} - {dataset.upper()}")
        print(80 * "-")
        trainer(embedding_name)
        print()


if __name__ == '__main__':
    for emb_name in sys.argv[1:]:
        print(80 * "=")
        print(emb_name)
        print(80 * "=")
        evaluate_embedding(emb_name)
