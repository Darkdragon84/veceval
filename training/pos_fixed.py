import sys

import veceval as ve
import numpy as np

from trainer import Trainer
from embedding_datasets import EmbeddingWindowCapsDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2

np.random.seed(ve.SEED)


class POSFixedTrainer(Trainer):
    TASK = ve.POS
    MODE = ve.FIXED

    def load_data(self):
        return EmbeddingWindowCapsDataset(self.train_data_path, self.embeddings,
                                          has_validation=True, is_testing=ve.IS_TESTING)

    def build_model(self):
        model = Sequential()
        model.add(Dense(input_shape=self.ds.X_train.shape[1:],
                        output_dim=ve.HIDDEN_SIZE,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.TANH))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                        output_dim=ve.POS_CLASSES,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.SOFTMAX))
        ve.compile_other_model(model, self.hp.optimizer)
        return model


def main():
    config_path, name = sys.argv[1:3]
    trainer = POSFixedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main()
