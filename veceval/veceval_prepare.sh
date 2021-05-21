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

for task in sentiment questions ner chunk nli pos
do
  echo "PREPARING DATA FOR $task"
  TRAIN_LOCATION=$ROOTDIR"/data/"$task"/scripts/"
  cd $TRAIN_LOCATION
  bash prepare_data.sh
  echo "===================================================================================="
done

