#!/usr/bin/env bash

# Teacher
MODEL_NAME=faster_rcnn_r50_c4_1x
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 905 \
                                    tools/train.py \
                                    --config configs/faster_rcnn/minicoco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/minicoco/$MODEL_NAME \
                                    --launcher pytorch


# Ablation
MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
for RATIO in 0.25 0.5 0.75
do
    for MIN_LR in 0.3 0.4 0.5 0.6 0.7
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 882 \
                                    tools/train.py \
                                    --config configs/faster_rcnn_kd/minicoco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/minicoco/$MODEL_NAME-$RATIO-$MIN_LR \
                                    --launcher pytorch \
                                    --cfg-options data.train.ratio_hr_lr_student=$RATIO data.train.min_lr_student=$MIN_LR
    done
done