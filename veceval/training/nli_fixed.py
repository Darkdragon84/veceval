import sys

import veceval.helpers.utility_functions as ve
import numpy as np

from veceval.training.trainer import Trainer
from veceval.training.embedding_datasets import EmbeddingPairDataset

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Merge
from keras.layers.recurrent import LSTM
from keras.regularizers import l2


class NLIFixedTrainer(Trainer):
    TASK = ve.NLI
    MODE = ve.FIXED

    def load_data(self):
        return EmbeddingPairDataset(self.train_data_path, self.embeddings,
                                    ve.NLI_MAX_LEN, has_validation=True,
                                    is_testing=ve.IS_TESTING)

    def build_model(self):
        premise = Sequential()
        premise.add(LSTM(input_shape=self.ds.X_p_train.shape[1:], output_dim=ve.HIDDEN_SIZE))
        premise.add(Dropout(ve.DROPOUT_PROB))

        hypothesis = Sequential()
        hypothesis.add(LSTM(input_shape=self.ds.X_h_train.shape[1:], output_dim=ve.HIDDEN_SIZE))
        hypothesis.add(Dropout(ve.DROPOUT_PROB))

        model = Sequential()
        model.add(Merge([premise, hypothesis], mode=ve.CONCAT))
        model.add(Dense(output_dim=ve.HIDDEN_SIZE, activation=ve.TANH, W_regularizer=l2(self.hp.dense_l2)))
        model.add(Dense(output_dim=ve.NLI_CLASSES, activation=ve.SOFTMAX, W_regularizer=l2(self.hp.dense_l2)))

        ve.compile_other_model(model, self.hp.optimizer)

        return model

    def train(self):
        callbacks = ve.callbacks(self.checkpoint_path, self.hp.stop_epochs)
        history = self.model.fit([self.ds.X_p_train, self.ds.X_h_train],
                                 self.ds.Y_train, batch_size=ve.BATCH_SIZE,
                                 nb_epoch=ve.MAX_EPOCHS, verbose=1,
                                 validation_data=([self.ds.X_p_val, self.ds.X_h_val],
                                                  self.ds.Y_val),
                                 callbacks=callbacks)

    def evaluate(self, set_to_evaluate=ve.VAL):
        if set_to_evaluate == ve.VAL:
            (X_p, X_h, Y) = self.ds.X_p_val, self.ds.X_h_val, self.ds.Y_val
        elif set_to_evaluate == ve.TRAIN:
            (X_p, X_h, Y) = self.ds.X_p_train, self.ds.X_h_train, self.ds.Y_train
        else:
            assert set_to_evaluate == ve.TEST and ve.TESTING == True
            (X_p, X_h, Y) = self.ds.X_p_test, self.ds.X_h_test, self.ds.Y_test

        _, acc = self.model.evaluate([X_p, X_h], Y)
        return set_to_evaluate, acc


def main_training(config_path, name):
    np.random.seed(ve.SEED)
    trainer = NLIFixedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main_training(*sys.argv[1:3])
