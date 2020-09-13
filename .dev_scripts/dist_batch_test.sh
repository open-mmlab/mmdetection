#!/usr/bin/env bash

MODEL_DIR=$1
JSON_OUT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    .dev_scripts/batch_test.py $MODEL_DIR $JSON_OUT --launcher pytorch ${@:4}
