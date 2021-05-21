import sys

import veceval.helpers.utility_functions as ve
import numpy as np

from veceval.training.trainer import Trainer
from veceval.training.index_datasets import IndexWindowCapsDataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Reshape, Merge

np.random.seed(ve.SEED)


class NERFinetunedTrainer(Trainer):
    TASK = ve.NER
    MODE = ve.FINE

    def load_data(self):
        return IndexWindowCapsDataset(self.train_data_path, self.embeddings, has_validation=True,
                                      has_caps=True, is_testing=ve.IS_TESTING)

    def build_model(self):

        vector_input = Sequential()
        vector_input.add(Embedding(input_dim=len(self.ds.vocab),
                                   output_dim=self.embedding_size,
                                   weights=[self.ds.weights],
                                   input_length=ve.WINDOW_SIZE))
        caps_input = Sequential()
        caps_input.add(Embedding(input_dim=ve.CAPS_DIMS,
                                 output_dim=ve.CAPS_DIMS,
                                 weights=[np.eye(ve.CAPS_DIMS)],
                                 input_length=ve.WINDOW_SIZE))
        model = Sequential()
        model.add(Merge([vector_input, caps_input], mode=ve.CONCAT))
        model.add(
            Reshape(((self.embedding_size + ve.CAPS_DIMS) * ve.WINDOW_SIZE,)))
        model.add(Dense(output_dim=ve.HIDDEN_SIZE))
        model.add(Activation(ve.TANH))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE, output_dim=ve.NER_CLASSES))
        model.add(Activation(ve.SOFTMAX))
        ve.compile_other_model(model, self.hp.optimizer)

        return model

    def train(self):
        callbacks = ve.callbacks(self.checkpoint_path, self.hp.stop_epochs)
        history = self.model.fit(
            list(self.ds.X_train), self.ds.Y_train, batch_size=ve.BATCH_SIZE,
            nb_epoch=ve.MAX_EPOCHS, verbose=1,
            validation_data=(list(self.ds.X_val), self.ds.Y_val),
            callbacks=callbacks)

    def evaluate(self, set_to_evaluate=ve.VAL):

        if set_to_evaluate == ve.VAL:
            (X, Y) = self.ds.X_val, self.ds.Y_val
        elif set_to_evaluate == ve.TRAIN:
            (X, Y) = self.ds.X_train, self.ds.Y_train
        else:
            assert set_to_evaluate == ve.TEST and ve.IS_TESTING == True
            (X, Y) = self.ds.X_test, self.ds.Y_test

        predictions = self.model.predict(list(X))
        result = ve.calculate_f1(predictions, Y)

        return set_to_evaluate, result


def main_training(config_path, name):
    trainer = NERFinetunedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main_training(*sys.argv[1:3])
