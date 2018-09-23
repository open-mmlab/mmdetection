#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON train.py $1 --dist --world-size $2 --rank 0 &
let MAX_RANK=$2-1
for i in `seq 1 $MAX_RANK`; do
    $PYTHON train.py $1 --dist --world-size $2 --rank $i > /dev/null 2>&1 &
done
