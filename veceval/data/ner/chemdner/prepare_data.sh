#!/bin/bash
for set in training evaluation validation
do
  unzip -o $set".csv.zip"
done

python make_datasets.py ./ ./ 5

for set in training evaluation validation
do
  rm $set".csv"
done