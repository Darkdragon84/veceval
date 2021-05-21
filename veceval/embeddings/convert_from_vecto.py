import os
import sys
import gzip
from pathlib import Path

from vecto.embeddings import load_from_dir

from veceval.embeddings.prepare_embeddings import SEP


def main():
    folder = Path(sys.argv[1])
    files = os.listdir(folder)
    for fname in files:
        fpath = folder / fname
        if fname.endswith(".txt") or fname.endswith(".gz"):
            os.remove(fpath)
            print(f"deleted {fpath}")

    embeddings = load_from_dir(folder)

    with gzip.open(folder / "vectors.txt.gz", "wt") as file:
        file.writelines(
            str(word) + SEP + SEP.join(
                [str(val) for val in embeddings.get_vector(word)]
            ) + "\n" for word in embeddings.vocabulary.lst_words
        )


if __name__ == '__main__':
    main()
