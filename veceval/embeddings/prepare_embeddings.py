import pickle

from vecto.embeddings import load_from_dir

import veceval.helpers.utility_functions as ve
import sys
import numpy as np


def main():
    embeddings_folder, common_vocab_file, output_file = sys.argv[1:]

    common_vocabulary = set()
    for line in open(common_vocab_file, 'r'):
        common_vocabulary.add(line.strip())

    embedding_dict = {}
    unk_vectors = []
    dim = None

    embeddings = load_from_dir(embeddings_folder)
    for word in embeddings.vocabulary.lst_words:
        word = word.lower()
        vector = embeddings.get_vector(word)
        dim = dim or len(vector)
        assert dim == len(vector)

        if word in common_vocabulary:
            embedding_dict[word] = vector
        else:
            unk_vectors.append(vector)

    embedding_dict[ve.PAD] = np.zeros(shape=(dim,))
    if unk_vectors:
        print(f"{len(unk_vectors)} words from vocabulary not in model")
        embedding_dict[ve.UNK] = sum(unk_vectors) / len(unk_vectors)
    else:
        embedding_dict[ve.UNK] = np.zeros(shape=(dim,))

    with open(output_file, 'wb') as output_file:
        pickle.dump(embedding_dict, output_file)


if __name__ == "__main__":
    main()
