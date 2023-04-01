# !/usr/bin/env bash

for MODEL_NAME in faster_rcnn_r50_c4_1x_mstrain_aug faster_rcnn_r50_fpn_3x_mstrain_aug
do
    CUDA_VISIBLE_DEVICES=1,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=899 \
                                        train.py \
                                        --config configs/faster_rcnn_msaug/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done