#!/bin/bash

# Error on a failed command
set -o errexit

# Get the root of the repository
ME="${BASH_SOURCE[0]}"
MYDIR=$( cd "$( dirname "$ME" )" && pwd )
export ROOTDIR="$MYDIR"

# Setup

export AFFILIATION="VecEval" # Ensure affiliation uses only a-z, A-Z, 0-9
if ! [[ $AFFILIATION =~ ^[A-Za-z]+$ ]]; then
  echo "Error: Please choose an affiliation using only letters."
  exit
fi

# Uncomment the line below and replace the embedding home path
#export EMBEDDINGS_HOME="/media/embeddings/" # directory containing gzipped embedding files
#export LOG_FILE=$ROOTDIR"/LOG"
#export CHECKPOINT_HOME="/tmp/veceval/checkpoints/"
#export PICKLES_HOME="/tmp/veceval/pickles/"
export LOG_FILE="/mnt/Data/iris_data/veceval/logs/log.txt"
export CHECKPOINT_HOME="/mnt/Data/iris_data/veceval/checkpoints"
export PICKLES_HOME="/mnt/Data/iris_data/veceval/pickles"
export EMBEDDING_NAME="News20_w2v"
#export EMBEDDING_NAME="wiki_cbow_50"

echo $ROOTDIR

# From here on, force all variables to be defined
set -o nounset


for task in nli pos sentiment questions ner chunk
do
  for with_backprop in fixed finetuned
  do
    echo ""
    echo "=================================================================================="
    echo $task $with_backprop
    echo "----------------------------------------------------------------------------------"
    TRAIN_SCRIPT=$ROOTDIR"/training/${task}_${with_backprop}.py"
    CONFIG_FILE=$ROOTDIR"/training/configs/config_${task}_${with_backprop}.txt"
    python ${TRAIN_SCRIPT} ${CONFIG_FILE} ${EMBEDDING_NAME}
    echo ""
  done
done
