#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
