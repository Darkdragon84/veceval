wget https://raw.githubusercontent.com/slavpetrov/universal-pos-tags/master/en-ptb.map
python preprocess_files.py en-ptb.map

cat en-ptb.map | cut -f2 | sort | uniq > tagset.txt
echo "O" >> tagset.txt

python make_datasets.py . ..  5

rm train.txt
rm dev.txt
rm en-ptb.map
