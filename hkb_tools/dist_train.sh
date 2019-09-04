#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
