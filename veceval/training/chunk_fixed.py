import sys

import veceval.helpers.utility_functions as ve
import numpy as np

from veceval.training.trainer import Trainer
from veceval.training.embedding_datasets import EmbeddingWindowCapsDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2

np.random.seed(ve.SEED)


class ChunkFixedTrainer(Trainer):
    TASK = ve.CHUNK
    MODE = ve.FIXED

    def build_model(self):
        model = Sequential()
        model.add(Dense(input_shape=self.ds.X_train.shape[1:],
                        output_dim=ve.HIDDEN_SIZE,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.TANH))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                        output_dim=ve.CHUNK_CLASSES,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.SOFTMAX))
        ve.compile_other_model(model, self.hp.optimizer)
        return model

    def load_data(self):
        return EmbeddingWindowCapsDataset(self.train_data_path, self.embeddings,
                                          has_validation=False, is_testing=ve.IS_TESTING)


def main_training(config_path, name):
    trainer = ChunkFixedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main_training(*sys.argv[1:3])
