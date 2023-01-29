# !/usr/bin/env bash

# MODEL_NAME=faster_rcnn_r50_fpn_1x_fskd
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 901 \
#                                     tools/train.py \
#                                     --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch


MODEL_NAME=faster_rcnn_r50_fpn_2x_fskd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 911 \
                                    tools/train.py \
                                    --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch


MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 905 \
                                    tools/train.py \
                                    --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch


MODEL_NAME=faster_rcnn_r50_c4_1x_mstrain_fskd
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 815 \
                                    train.py \
                                    --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch