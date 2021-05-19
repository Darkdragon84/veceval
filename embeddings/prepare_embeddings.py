import pickle
import veceval as ve
import sys
import gzip
import numpy as np

DIM = 50
SEP = "\t"


def main():
    embeddings_file, common_vocab_file, output_file = sys.argv[1:]

    common_vocabulary = set()
    for line in open(common_vocab_file, 'r'):
        common_vocabulary.add(line.strip())

    embedding_dict = {}
    unk_vectors = []

    with gzip.open(embeddings_file, 'r') as embedding_file:
        for line in embedding_file:
            this_line = line.decode().split(SEP)
            assert len(this_line) == DIM + 1  # does this mean we can only use dim = 50?
            word = this_line[0].lower()
            vector = np.array([float(x) for x in this_line[1:]])
            if word in common_vocabulary:
                embedding_dict[word] = vector
            else:
                unk_vectors.append(vector)

    embedding_dict[ve.PAD] = np.zeros(shape=(DIM,))
    if unk_vectors:
        embedding_dict[ve.UNK] = sum(unk_vectors) / len(unk_vectors)
    else:
        embedding_dict[ve.UNK] = np.zeros(shape=(DIM,))

    with open(output_file, 'wb') as output_file:
        pickle.dump(embedding_dict, output_file)


if __name__ == "__main__":
    main()
