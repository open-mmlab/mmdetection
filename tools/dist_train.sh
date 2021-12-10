#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --standalone --nnodes=1 --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG ${@:3}
