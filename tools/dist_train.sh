#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=`shuf -i 2000-3000 -n 1`

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
