#!/bin/bash
set -o errexit

ROOT_DIR=$PWD
for dataset in conll chemdner
do
  cd $ROOT_DIR"/"$dataset
  bash prepare_data.sh
done
