#!/usr/bin/env bash

# Teacher
# MODEL_NAME=faster_rcnn_r50_c4_1x
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 905 \
#                                     tools/train.py \
#                                     --config configs/faster_rcnn/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch


# Ablation
# MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
# for RATIO in 0.5
# do
#     for MIN_LR in 0.7 0.8
#     do
#         CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 822 \
#                                     train.py \
#                                     --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco_ablation/$MODEL_NAME-$RATIO-$MIN_LR \
#                                     --launcher pytorch \
#                                     --cfg-options data.train.ratio_hr_lr_student=$RATIO data.train.min_lr_student=$MIN_LR
#     done
# done



# # RATIO
# MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
# for RATIO in 0.25 0.75
# do
#     for MIN_LR in 0.5
#     do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 822 \
#                                     train.py \
#                                     --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco_ablation/$MODEL_NAME-$RATIO-$MIN_LR \
#                                     --launcher pytorch \
#                                     --cfg-options data.train.ratio_hr_lr_student=$RATIO data.train.min_lr_student=$MIN_LR
#     done
# done


# RATIO
MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
for RATIO in 0.5
do
    for MIN_LR in 0.6
    do
        for D_PARAM in 0.5 1.0 1.5 2.0
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port 822 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco_ablation/$MODEL_NAME-param$D_PARAM \
                                        --launcher pytorch \
                                        --cfg-options data.train.ratio_hr_lr_student=$RATIO data.train.min_lr_student=$MIN_LR model.distill_param=$D_PARAM
        done
    done
done



MODEL_NAME=faster_rcnn_r50_c4_1x_fskd
for RATIO in 0.5
do
    for MIN_LR in 0.6
    do
        for D_PARAM in 2.5 3.0 3.5 4.0
        do
            CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port 822 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco_ablation/$MODEL_NAME-param$D_PARAM \
                                        --launcher pytorch \
                                        --cfg-options data.train.ratio_hr_lr_student=$RATIO data.train.min_lr_student=$MIN_LR model.distill_param=$D_PARAM
        done
    done
done


# # MSTRAIN Teacher
# MODEL_NAME=faster_rcnn_r50_c4_1x_mstrain
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 901 \
#                                     train.py \
#                                     --config configs/faster_rcnn/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco_ablation/$MODEL_NAME \
#                                     --launcher pytorch


# # NAIVE
# MODEL_NAME=faster_rcnn_r50_c4_1x_naive
# CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 905 \
#                                     train.py \
#                                     --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco_ablation/$MODEL_NAME \
#                                     --launcher pytorch



# BOTH (POS + NEG)
# MODEL_NAME=faster_rcnn_r50_c4_1x_fskd_both
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 112 \
#                                     train.py \
#                                     --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco_ablation/$MODEL_NAME \
#                                     --launcher pytorch \
#                                     --cfg-options model.distill_type=both