#!/usr/bin/env bash
MODEL_NAME=faster_rcnn_r50_fpn_1x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 901 \
                                    tools/train.py \
                                    --config configs/faster_rcnn/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch


MODEL_NAME=faster_rcnn_r50_c4_1x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 905 \
                                    tools/train.py \
                                    --config configs/faster_rcnn/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch


MODEL_NAME=faster_rcnn_r50_fpn_2x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 911 \
                                    tools/train.py \
                                    --config configs/faster_rcnn/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch