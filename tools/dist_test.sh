#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT ${@:4}
