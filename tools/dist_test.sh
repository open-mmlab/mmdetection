#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

# CUDA_VISIBLE_DEVICES=2,3 
# ./tools/dist_test.sh \ 
#     configs/_HRSID_detection/faster_rcnn_r50_fpn_2x_HRSID.py \
#     work_dirs/faster_rcnn_r50_fpn_2x_HRSID/best_bbox_mAP_epoch_20.pth \
#     2 \
#     --eval bbox \
#     --out work_dirs/faster_rcnn_r50_fpn_2x_HRSID/results.pkl \
#     --options "classwise=True" \
#     2 > log.txt 2>&1