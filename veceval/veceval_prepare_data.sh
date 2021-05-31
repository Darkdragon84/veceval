#!/bin/bash

# Error on a failed command
set -o errexit

# Get the root of the repository
ME="${BASH_SOURCE[0]}"
MYDIR=$( cd "$( dirname "$ME" )" && pwd )
export ROOTDIR="$MYDIR"

echo $ROOTDIR

# From here on, force all variables to be defined
set -o nounset

for task in sentiment questions  chunk nli pos ner
do
  echo "===================================================================================="
  echo "PREPARING DATA FOR $task"
  TRAIN_LOCATION=$ROOTDIR"/data/"$task
  cd $TRAIN_LOCATION
  echo $PWD
  echo "===================================================================================="
  bash prepare_data.sh
done

