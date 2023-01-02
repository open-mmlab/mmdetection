#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
#                                 --nproc_per_node=2 \
#                                 tools/train.py \
#                                 configs/faster_rcnn_cont/faster_rcnn_r50_caffe_c4_1x_coco.py \
#                                 --seed 0 \
#                                 --work-dir result/faster_rcnn_cont \
#                                 --launcher pytorch


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
                                --nproc_per_node=2 \
                                --master_port 112 \
                                tools/train.py \
                                configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco.py \
                                --seed 0 \
                                --work-dir result/faster_rcnn_ori \
                                --launcher pytorch