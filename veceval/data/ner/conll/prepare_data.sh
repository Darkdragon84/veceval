#!/bin/bash
wget -O conll2003.zip https://data.deepai.org/conll2003.zip
unzip conll2003.zip -d conll2003
mv conll2003/train.txt ./
mv conll2003/valid.txt ./
rm -r conll2003

python make_datasets.py ./ ./ 5

rm train.txt
rm valid.txt
rm conll2003.zip
