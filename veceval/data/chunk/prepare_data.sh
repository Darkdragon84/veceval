wget -O train.txt.gz https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
gunzip train.txt.gz

python make_datasets.py ./ ./ 5

rm train.txt