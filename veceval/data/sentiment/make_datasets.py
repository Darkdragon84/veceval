# -*- coding: utf-8 -*-
import sys
import codecs

from veceval.data import data_lib as dl

from collections import defaultdict
from math import ceil


def read_dataset_sentences(input_prefix):
    id_to_sentence = {}

    with open(input_prefix + '/datasetSentences.txt', 'r', encoding='utf-8') as dataset_sentences_file:

    # dataset_sentences_file = codecs.open(
    #     input_prefix + '/datasetSentences.txt', 'r', 'utf-8')

        # dataset_sentences : sentence index -> sentence string
        dataset_sentences_file.readline()

        for line in dataset_sentences_file:
            (sent_id, sentence) = line.split('\t')
            # there are some encoding issues in the datasetSentences txt file
            id_to_sentence[sent_id] = sentence.strip().encode('latin-1').decode('utf-8')
            # id_to_sentence[sent_id] = sentence[:-1].encode('latin-1').decode('utf-8')

    return id_to_sentence


def read_dictionary(input_prefix):
    # dictionary : phrase -> phrase id
    dictionary = {}
    # dictionary_file = codecs.open(input_prefix + '/dictionary.txt', 'r', 'utf-8')
    with open(input_prefix + '/dictionary.txt', 'r', encoding='utf-8') as dictionary_file:
        for line in dictionary_file:
            (phrase, phrase_id) = line.split('|')
            # dictionary[phrase] = phrase_id[:-1]
            dictionary[phrase] = phrase_id.strip()

    return dictionary


def read_sentiment_labels(input_prefix):
    # sentiment_labels : phrase id -> sentiment
    phrase_to_label = {}

    with open(input_prefix + '/sentiment_labels.txt', 'r', encoding="us-ascii") as sentiment_labels_file:

        sentiment_labels_file.readline()
        for line in sentiment_labels_file:
            (phrase_id, sentiment) = line.split('|')
            phrase_to_label[phrase_id] = sentiment.strip()


    # sentiment_labels_file = codecs.open(
    #     input_prefix + '/sentiment_labels.txt', 'r', 'utf-8')
    #
    # sentiment_labels_file.readline()
    #
    # for line in sentiment_labels_file:
    #     (phrase_id, sentiment) = line.split('|')
    #     phrase_to_label[phrase_id] = sentiment[:-1]

    return phrase_to_label


def read_dataset_split(input_prefix):
    # dataset_split : sentence index -> sentence set
    dataset_split = defaultdict(list)
    dataset_split_file = codecs.open(
        input_prefix + '/datasetSplit.txt', 'r', 'utf-8')

    dataset_split_file.readline()

    for line in dataset_split_file:
        (sent_id, sent_set) = line.split(',')
        dataset_split[sent_set[:-1]].append(sent_id)

    # 1 = train ; 2 = test ; 3 = dev
    return dataset_split[u"1"], dataset_split[u"2"], dataset_split[u"3"]


# def get_binary_label(binary_label_map, string_label):
#     label = binary_label_map[ceil(5.0 * float(string_label))]
#     if label is not None:
#         return str(label).encode("utf-8")
#     else:
#         return label


def construct_sentiment_dataset(input_prefix):
    # binary_label_map = {0.0: 0.0,
    #                     1.0: 0.0,
    #                     2.0: 0.0,
    #                     3.0: None,
    #                     4.0: 1.0,
    #                     5.0: 1.0}
    labels = ["0", "1"]
    label_map = dl.make_label_map(labels)

    dataset_sentences = read_dataset_sentences(input_prefix)
    dictionary = read_dictionary(input_prefix)
    sentiment_labels = read_sentiment_labels(input_prefix)
    train, test, dev = read_dataset_split(input_prefix)
    new_train = []
    new_test = []
    new_dev = []
    for old, new in zip([train, test, dev], [new_train, new_test, new_dev]):
        # temp = []
        # labels = set()
        for sent_id in old:
            phrase = dataset_sentences[sent_id]
            phrase = phrase.replace("-LRB-", "(").replace("-RRB-", ")")
            string_label = sentiment_labels[dictionary[phrase]]
            # binary_sentiment_label = get_binary_label(binary_label_map, string_label)
            # TODO make undecided category in the middle
            binary_sentiment_label = labels[0] if float(string_label) < 0.5 else labels[1]
            if binary_sentiment_label is not None:
                new.append((
                    [word.lower() for word in phrase.split()],
                    label_map[binary_sentiment_label]
                ))
        # labels.add(binary_sentiment_label)
        # label_map = dl.make_label_map(sorted(list(labels)))
        # for phrase, label in temp:
        #     lowered_phrase = [word.lower() for word in phrase]
        #     new.append((lowered_phrase, label_map[label]))
        #         if binary_sentiment_label is not None:
        #             temp.append((phrase.split(), binary_sentiment_label))
                    # labels.add(binary_sentiment_label)
            # label_map = dl.make_label_map(sorted(list(labels)))
            # for phrase, label in temp:
            #     lowered_phrase = [word.lower() for word in phrase]
            #     new.append((lowered_phrase, label_map[label]))

    return new_train, new_test, new_dev, label_map


def main():
    input_prefix, output_prefix = sys.argv[1:]

    train, test, dev, label_map = construct_sentiment_dataset(input_prefix)
    for dataset, filename in zip([train, dev], ["train", "dev"]):
        out_file = output_prefix + filename + ".pickle"
        dl.make_pickle(dataset, label_map, out_file)


if __name__ == "__main__":
    main()
