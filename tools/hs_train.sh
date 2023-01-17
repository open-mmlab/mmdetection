

MODEL_NAME='coco_faster_rcnn_r50_caffe_dc5_1x'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 1025 \
                                train.py \
                                --config configs/faster_rcnn/${MODEL_NAME}.py \
                                --seed 0 \
                                --work-dir /ailab_mat/checkpoints/sung/${MODEL_NAME} \
                                --launcher pytorch

MODEL_NAME='coco_faster_rcnn_r101_fpn_1x'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 1025 \
                                train.py \
                                --config configs/faster_rcnn/${MODEL_NAME}.py \
                                --seed 0 \
                                --work-dir /ailab_mat/checkpoints/sung/${MODEL_NAME} \
                                --launcher pytorch

# MODEL_NAME='coco_faster_rcnn_r101_fpn_2x'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                 --nproc_per_node=4 \
#                                 --master_port 1025 \
#                                 train.py \
#                                 --config configs/faster_rcnn/${MODEL_NAME}.py \
#                                 --seed 0 \
#                                 --work-dir /ailab_mat/checkpoints/sung/${MODEL_NAME} \
#                                 --launcher pytorch

# MODEL_NAME='coco_faster_rcnn_x101_32x4d_fpn_1x'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                 --nproc_per_node=4 \
#                                 --master_port 1025 \
#                                 train.py \
#                                 --config configs/faster_rcnn/${MODEL_NAME}.py \
#                                 --seed 0 \
#                                 --work-dir /ailab_mat/checkpoints/sung/${MODEL_NAME} \
#                                 --launcher pytorch
