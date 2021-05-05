#!/bin/bash

START_ID=854
DATASET_ID=816
for i in $(seq 1 10);
do
    python train.py $(($START_ID+$i)) 1 3 $DATASET_ID $i
done

