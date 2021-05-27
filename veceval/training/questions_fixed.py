import sys
import veceval.helpers.utility_functions as ve
import numpy as np

from veceval.training.trainer import Trainer
from veceval.training.embedding_datasets import EmbeddingDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2


class QuestionsFixedTrainer(Trainer):
    TASK = ve.QUESTIONS
    MODE = ve.FIXED

    def build_model(self):
        model = Sequential()
        model.add(
            LSTM(input_shape=self.ds.X_train.shape[1:], output_dim=ve.HIDDEN_SIZE))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE,
                        output_dim=ve.QUESTIONS_CLASSES,
                        W_regularizer=l2(self.hp.dense_l2)))
        model.add(Activation(ve.SOFTMAX))
        ve.compile_other_model(model, self.hp.optimizer)
        return model

    def load_data(self):
        return EmbeddingDataset(self.train_data_path, self.embeddings,
                                ve.QUESTIONS_MAX_LEN, has_validation=False,
                                is_testing=ve.IS_TESTING)


def main_training(config_path, name):
    np.random.seed(ve.SEED)
    trainer = QuestionsFixedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main_training(*sys.argv[1:3])
