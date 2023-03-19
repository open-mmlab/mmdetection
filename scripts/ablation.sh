#!/usr/bin/env bash
MODEL_NAME=faster_rcnn_r101_fpn_1x_mstrain_fskd
for min_LR in 0.2 0.4 0.6 0.8
do
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                            --nproc_per_node=4 \
                            --master_port 801 \
                            train.py \
                            --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                            --seed 0 \
                                --work-dir result/test/coco/$MODEL_NAME-backbone_1.0-rpn_1.0-minLR_$min_LR \
                                --launcher pytorch \
                                --cfg-options model.distill_param=1.0 model.distill_param_backbone=1.0 data.train.min_lr_student=$min_LR
done



MODEL_NAME=faster_rcnn_r101_fpn_1x_mstrain_fskd
for BACKBONE in 0.5 1.0 1.5 2.0
do
    for RPN in 0.5 1.0
    do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 799 \
                                train.py \
                                --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                --seed 0 \
                                    --work-dir result/test/coco/$MODEL_NAME-backbone_$BACKBONE-rpn_$RPN \
                                    --launcher pytorch \
                                    --cfg-options model.distill_param=$RPN model.distill_param_backbone=$BACKBONE
    done
done



MODEL_NAME=faster_rcnn_r101_fpn_1x_mstrain_fskd
for BACKBONE in 0.5 1.0 1.5 2.0
do
    for RPN in 1.5 2.0
    do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 852 \
                                train.py \
                                --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                --seed 0 \
                                    --work-dir result/test/coco/$MODEL_NAME-backbone_$BACKBONE-rpn_$RPN \
                                    --launcher pytorch \
                                    --cfg-options model.distill_param=$RPN model.distill_param_backbone=$BACKBONE
    done
done













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