import os
import sys
from functools import partial

from veceval.settings import CONFIGS_FOLDER, MODES, TASK_TO_DATASET
import importlib


def compile_trainers():
    task_to_dataset_to_trainer = {}
    for task, datasets in TASK_TO_DATASET.items():
        for mode in MODES:
            base_name = f"{task}_{mode}"
            config_file_path = CONFIGS_FOLDER / f"config_{base_name}.txt"
            if not os.path.isfile(config_file_path):
                continue
            dataset_to_trainer = {}
            for dataset in datasets:
                try:
                    trainer = getattr(importlib.import_module(f"veceval.training.{base_name}"), "main_training")
                    trainer = partial(trainer, config_file_path, dataset)
                except (ImportError, ModuleNotFoundError, AttributeError):
                    continue
                dataset_to_trainer[dataset] = trainer
            task_to_dataset_to_trainer[base_name] = dataset_to_trainer
    return task_to_dataset_to_trainer


def evaluate_embedding(embedding_name):
    task_to_trainer = compile_trainers()
    for task_name, dataset_to_trainer in task_to_trainer.items():
        for dataset, trainer in dataset_to_trainer.items():
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
