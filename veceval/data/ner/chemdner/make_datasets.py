import sys

import pandas

from veceval.data import data_lib as dl


def calculate_pad_length(window_size):
    assert window_size % 2 == 1
    return int((window_size - 1) / 2)


def read_dataset_sentences(input_file, window_size):
    pad_length = calculate_pad_length(window_size)
    padding = [(None, None)] * pad_length
    dataset = list(padding)
    df = pandas.read_csv(input_file, na_filter=False)

    sentences = df.groupby(['PMID_Type', 'Sentence_Index'])
    for _, sentence in sentences:
        dataset += list(zip(sentence.Token, sentence.Tag))
        dataset += list(padding)

    dataset += list(padding)
    return dataset


def make_windows(dataset, window_size):
    windows = []
    pad_length = calculate_pad_length(window_size)
    for i, (word, label) in enumerate(dataset):
        if word is not None:
            window = [dataset[j][0] or "PAD" for j in range(i - pad_length,
                                                            i + pad_length + 1)]
            windows.append((window, label))
    return windows


def main():
    input_prefix, output_prefix, window_size = sys.argv[1:]
    window_size = int(window_size)

    train = read_dataset_sentences(input_prefix + '/training.csv', window_size)
    valid = read_dataset_sentences(input_prefix + '/validation.csv', window_size)
    train_windows = make_windows(train, window_size)
    valid_windows = make_windows(valid, window_size)

    labels = set()
    for dataset in [train_windows, valid_windows]:
        for _, label in dataset:
            labels.add(label)

    labels = sorted(list(labels))
    label_map = dl.make_label_map(labels)

    for dataset, filename in zip([train_windows, valid_windows],
                                 ["train", "dev"]):
        out_file = output_prefix + filename + ".pickle"
        new_dataset = [(window, label_map[label]) for window, label in dataset]
        dl.make_pickle(new_dataset, label_map, out_file)


if __name__ == "__main__":
    main()
