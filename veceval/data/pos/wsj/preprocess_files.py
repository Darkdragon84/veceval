import sys
from nltk.corpus import treebank

NONE_TAG = "-NONE-"


def make_pos_map(path):
    pos_map = {}
    for line in open(path, 'r'):
        (tag, u_tag) = line.split()
        if '|' not in tag:
            pos_map[tag] = u_tag
    return pos_map


def main():
    POS_map = make_pos_map(sys.argv[1])

    train_fileids = treebank.fileids()[:18]
    dev_fileids = treebank.fileids()[18:21]
    for out_name, in_name in zip(["train.txt", "dev.txt"],
                                 [train_fileids, dev_fileids]):
        with open(out_name, 'w') as out_file:
            for word, tag in treebank.tagged_words(in_name):
                if tag == NONE_TAG:
                    continue
                out_file.write(word.lower() + "\t" + POS_map[tag] + "\n")
            out_file.write("\n")


if __name__ == "__main__":
    main()
