import sys

import veceval as ve
import numpy as np

from trainer import Trainer
from embedding_datasets import EmbeddingDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

np.random.seed(ve.SEED)


class SentimentFixedTrainer(Trainer):
    TASK = ve.SENTIMENT
    MODE = ve.FIXED

    def load_data(self):
        return EmbeddingDataset(self.train_data_path, self.embeddings,
                                ve.SENTIMENT_MAX_LEN, has_validation=True,
                                is_testing=ve.IS_TESTING)

    def build_model(self):
        model = Sequential()
        model.add(
            LSTM(input_shape=self.ds.X_train.shape[1:], output_dim=ve.HIDDEN_SIZE))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                        output_dim=ve.SENTIMENT_CLASSES,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.SIGMOID))
        ve.compile_binary_model(model, self.hp.optimizer)
        return model


def main():
    config_path, name = sys.argv[1:3]
    trainer = SentimentFixedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main()
