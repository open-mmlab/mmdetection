#!/usr/bin/env bash

# RATIO
for D_PARAM in 0.0 1.0
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                            --nproc_per_node=4 \
                            --master_port 822 \
                            train.py \
                            --config configs/test/coco_r50_fpn_1x_mstrain_test.py \
                            --seed 0 \
                                --work-dir result/test/r50_fpn_1x_mstrain_test_backbone1.0-rpn$D_PARAM \
                                --launcher pytorch \
                                --cfg-options model.distill_param=$D_PARAM
done



for D_PARAM in 0.0 1.0
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                            --nproc_per_node=4 \
                            --master_port 823 \
                            train.py \
                            --config configs/test/coco_r101_fpn_1x_mstrain_test.py \
                            --seed 0 \
                                --work-dir result/test/r101_fpn_1x_mstrain_test_backbone1.0-rpn$D_PARAM \
                                --launcher pytorch \
                                --cfg-options model.distill_param=$D_PARAM
done