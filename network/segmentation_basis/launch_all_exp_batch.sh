#!/bin/bash

START_ID=1236
DATASET_ID=816
Array=(1 10 100 1000 10000 100000)
for i in "${!Array[@]}";
#for i in $(seq 1 10);
do
    #python train.py $(($START_ID+$i)) 3 3 $DATASET_ID $i 64
    python train.py $(($START_ID+$i)) 3 3 $DATASET_ID ${Array[$i]} 64 0
done

