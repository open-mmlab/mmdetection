#!/usr/bin/env bash

set -x

JOB_NAME=debug
PARTITION=$1
CONFIG=$2
CKPT=$3
OUTFILE=$4
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-40}
SRUN_ARGS=${SRUN_ARGS:-""}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/deployment/pytorch2onnx.py ${CONFIG} ${CKPT} \
    --output-file ${OUTFILE} \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 608 608 \
    --mean 0 0 0 \
    --std 255 255 255 \
    --show \
    --verify \
    --dynamic-export \

# default 608
~                 
