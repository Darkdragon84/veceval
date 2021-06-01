#!/bin/bash

# Error on a failed command
set -o errexit

# Get the directory of current script
ME="${BASH_SOURCE[0]}"
SCRIPTDIR=$( cd "$( dirname "$ME" )" && pwd )

echo "$SCRIPTDIR"
# From here on, force all variables to be defined
set -o nounset

rm -Rf "__pycache__"
#for task in sentiment questions chunk nli pos ner
for task in */
do
  echo "===================================================================================="
  echo "PREPARING DATA FOR $task"
  echo "===================================================================================="
  TASK_LOCATION="$SCRIPTDIR/$task"
  cd "$TASK_LOCATION"
  echo "$PWD"

  rm -Rf "__pycache__"
  for dataset in */
  do
    cd "$TASK_LOCATION/$dataset"
    echo "--------------------------------------------------------------------------------------------------"
    echo "PREPARING $dataset DATA"
    echo "--------------------------------------------------------------------------------------------------"
    bash prepare_data.sh || true  # continue despite failure
    rm -Rf "__pycache__"
  done
  rm -Rf "__pycache__"
done

