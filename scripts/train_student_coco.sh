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
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m torch.distributed.launch \
                                    --nproc_per_node=4 \
                                    --master_port 815 \
                                    train.py \
                                    --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                    --seed 0 \
                                    --work-dir result/coco/$MODEL_NAME \
                                    --launcher pytorch




for MODEL_NAME in faster_rcnn_r50_fpn_3x_fskd faster_rcnn_r101_fpn_3x_fskd faster_rcnn_x101_32x4d_fpn_2x_fskd faster_rcnn_x101_32x4d_fpn_3x_fskd faster_rcnn_x101_64x4d_fpn_1x_fskd faster_rcnn_x101_64x4d_fpn_2x_fskd faster_rcnn_x101_64x4d_fpn_3x_fskd
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=912 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done


for MODEL_NAME in faster_rcnn_r101_fpn_3x_mstrain_fskd faster_rcnn_x101_32x4d_fpn_3x_mstrain_fskd
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=899 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done


for MODEL_NAME in faster_rcnn_x101_64x4d_fpn_3x_mstrain_fskd
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=899 \
                                        train.py \
                                        --config configs/faster_rcnn_kd/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done