#!/bin/bash
wget -O train.label http://cogcomp.org/Data/QA/QC/train_5500.label
python make_datasets.py . ./

rm train.label
