#!/usr/bin/env bash
MODEL_NAME=faster_rcnn_r50_c4_1x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 905 \
                                    tools/train.py \
                                    --config configs/faster_rcnn/minicoco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/minicoco/$MODEL_NAME \
                                    --launcher pytorch
