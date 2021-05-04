#!/usr/bin/env bash

set -x

JOB_NAME=debug
EVALUATION_METRICS='bbox'
PARTITION=$1
CONFIG=$2
ONNX_FILE=$3
OUTPUT_FILE=$4
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
    python tools/deployment/test.py \
    ${CONFIG} \
    ${ONNX_FILE} \
    --out ${OUTPUT_FILE} \
    --eval ${EVALUATION_METRICS} \
    # --show-dir ${SHOW_DIRECTORY} \
    # ----show-score-thr ${SHOW_SCORE_THRESHOLD} \
    # ----cfg-options ${CFG_OPTIONS} \
    # --format-only ${FORMAT_ONLY} \
    # ----eval-options ${EVALUATION_OPTIONS} \
