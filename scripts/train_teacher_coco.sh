# !/usr/bin/env bash
# MODEL_NAME=faster_rcnn_r50_fpn_1x
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 901 \
#                                     tools/train.py \
#                                     --config configs/faster_rcnn/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch


# MODEL_NAME=faster_rcnn_r50_c4_1x
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 905 \
#                                     tools/train.py \
#                                     --config configs/faster_rcnn/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch


# MODEL_NAME=faster_rcnn_r50_fpn_2x
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 911 \
#                                     tools/train.py \
#                                     --config configs/faster_rcnn/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch



for MODEL_NAME in faster_rcnn_r50_dc5_1x_mstrain faster_rcnn_r50_fpn_1x_mstrain faster_rcnn_r50_fpn_2x_mstrain faster_rcnn_r101_fpn_1x_mstrain faster_rcnn_r101_fpn_2x_mstrain faster_rcnn_x101_32x4d_fpn_1x_mstrain faster_rcnn_x101_32x4d_fpn_2x_mstrain faster_rcnn_x101_64x4d_fpn_1x_mstrain faster_rcnn_x101_64x4d_fpn_2x_mstrain
do
    CUDA_VISIBLE_DEVICES=1,5,6,7 python -m torch.distributed.launch \
                                        --nproc_per_node=4 \
                                        --master_port=911 \
                                        train.py \
                                        --config configs/faster_rcnn/coco_$MODEL_NAME.py \
                                        --seed 0 \
                                        --work-dir result/coco/$MODEL_NAME \
                                        --launcher pytorch
done








# ## FCOS settings
# MODEL_NAME=fcos_r50_fpn_gn-head_1x
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 911 \
#                                     train.py \
#                                     --config configs/fcos/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch


# MODEL_NAME=fcos_tricks_r50_fpn_gn-head_1x
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#                                     --nproc_per_node=4 \
#                                     --master_port 902 \
#                                     train.py \
#                                     --config configs/fcos/coco_$MODEL_NAME.py \
#                                     --seed 0 \
#                                     --work-dir result/coco/$MODEL_NAME \
#                                     --launcher pytorch