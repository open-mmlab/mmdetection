#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
#                                 --nproc_per_node=2 \
#                                 --master_port 203 \
#                                 tools/train.py \
#                                 --config configs/faster_rcnn_kd/faster_rcnn_r50_c4_1x_range.py \
#                                 --seed 0 \
#                                 --work-dir result/faster_rcnn_range \
#                                 --launcher pytorch


# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
#                                 --nproc_per_node=2 \
#                                 --master_port 101 \
#                                 tools/train.py \
#                                 --config configs/faster_rcnn_kd/faster_rcnn_r50_c4_1x_value_equal.py \
#                                 --seed 0 \
#                                 --work-dir result/faster_rcnn_value_equal \
#                                 --launcher pytorch


CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
                                --nproc_per_node=2 \
                                --master_port 901 \
                                train.py \
                                --config configs/faster_rcnn/smdp_faster_rcnn_r50_caffe_c4_1x.py \
                                --seed 0 \
                                --work-dir result/smdp/faster_rcnn_r50_c4_1x_ori \
                                --launcher pytorch


CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
                                --nproc_per_node=2 \
                                --master_port 911 \
                                train.py \
                                --config configs/faster_rcnn_kd/smdp_faster_rcnn_r50_c4_1x_discrete.py \
                                --seed 0 \
                                --work-dir result/smdp/faster_rcnn_r50_c4_1x_discrete \
                                --launcher pytorch


# # Single GPU
# python train.py --gpu-id 7 \
#                 --config configs/faster_rcnn/smdp_faster_rcnn_r50_caffe_c4_1x.py \
#                 --seed 0 \
#                 --work-dir result/smdp/faster_rcnn_r50_c4_1x_ori