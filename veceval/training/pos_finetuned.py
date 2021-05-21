import sys

import veceval.training.veceval as ve
import numpy as np

from veceval.training.trainer import Trainer
from veceval.training.index_datasets import IndexWindowCapsDataset

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Activation, Reshape

np.random.seed(ve.SEED)


class POSFinetunedTrainer(Trainer):
    TASK = ve.POS
    MODE = ve.FINE

    def load_data(self):
        return IndexWindowCapsDataset(self.train_data_path, self.embeddings,
                                      has_validation=True, is_testing=ve.IS_TESTING)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=len(self.ds.vocab),
                            output_dim=self.embedding_size,
                            weights=[self.ds.weights],
                            input_length=ve.WINDOW_SIZE))
        model.add(Reshape((self.embedding_size * ve.WINDOW_SIZE,)))
        model.add(Dense(output_dim=ve.HIDDEN_SIZE))
        model.add(Activation(ve.TANH))
        model.add(Dropout(ve.DROPOUT_PROB))
        model.add(Dense(input_dim=ve.HIDDEN_SIZE, output_dim=ve.POS_CLASSES))
        model.add(Activation(ve.SOFTMAX))
        ve.compile_other_model(model, self.hp.optimizer)
        return model


def main():
    config_path, name = sys.argv[1:3]
    trainer = POSFinetunedTrainer(config_path, name)
    trainer.train_and_test()


if __name__ == "__main__":
    main()
