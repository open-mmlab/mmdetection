#!/usr/bin/env bash

# RATIO
MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
D_PARAM=0.5
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                            --nproc_per_node=4 \
                            --master_port 822 \
                            train.py \
                            --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                            --seed 0 \
                                --work-dir result/coco_ablation/$MODEL_NAME-kd_param$D_PARAM \
                                --launcher pytorch \
                                --cfg-options model.distill_param_kd=$D_PARAM
