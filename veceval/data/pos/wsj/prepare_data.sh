#!/bin/bash
# simplified POS tag mapping
wget -O en-ptb.map https://raw.githubusercontent.com/slavpetrov/universal-pos-tags/master/en-ptb.map

python preprocess_files.py en-ptb.map

# not sure we need this?
#cat en-ptb.map | cut -f2 | sort | uniq > tagset.txt
#echo "O" >> tagset.txt

python make_datasets.py ./ ./  5

rm en-ptb.map
rm train.txt
rm dev.txt
