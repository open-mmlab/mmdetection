#!/usr/bin/env bash

# Ablation
MODEL_NAME=fcos_r50_fpn_gn-head_1x_fskd
for D_TYPE in both reg cls
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 822 \
                                train.py \
                                --config configs/fcos_kd/coco_$MODEL_NAME.py \
                                --seed 0 \
                                --work-dir result/fcos_ablation/$MODEL_NAME-$D_TYPE \
                                --launcher pytorch \
                                --cfg-options model.distill_type=$D_TYPE
done