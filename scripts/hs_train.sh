
# # #--------------------------------   FASTER RCNN TEACHER   --------------------------------#

# MODEL_NAME='faster_rcnn_r50_dc5_1x'
# MODEL_NAME='faster_rcnn_r101_fpn_1x'
# MODEL_NAME='faster_rcnn_r101_fpn_2x'
# MODEL_NAME='faster_rcnn_x101_32x4d_fpn_1x'
# MODEL_NAME='faster_rcnn_r50_fpn_2x'
# FOLDER_NAME='faster_rcnn'

# # #--------------------------------   FASTER RCNN STUDENT   --------------------------------#

# MODEL_NAME='faster_rcnn_r50_dc5_1x_fskd'
# FOLDER_NAME='faster_rcnn_kd'

# # #--------------------------------   MASK RCNN TEACHER   --------------------------------#

# MODEL_NAME='mask_rcnn_r50_fpn_1x'
# MODEL_NAME='mask_rcnn_r101_fpn_1x'
# FOLDER_NAME='mask_rcnn'

# # #--------------------------------   MASK RCNN STUDENT   --------------------------------#

# MODEL_NAME='mask_rcnn_r50_fpn_1x_fskd'
# MODEL_NAME='mask_rcnn_r101_fpn_1x_fskd'
# FOLDER_NAME='mask_rcnn_kd'

# #--------------------------------   SPARSE RCNN TEACHER   --------------------------------#

# MODEL_NAME='sparse_rcnn_r101_fpn_mstrain_480-800_3x'
# FOLDER_NAME='sparse_rcnn'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#                                 --nproc_per_node=4 \
#                                 --master_port 1028 \
#                                 train.py \
#                                 --config configs/${FOLDER_NAME}/coco_${MODEL_NAME}.py \
#                                 --seed 0 \
#                                 --work-dir result/coco/${MODEL_NAME} \
#                                 --launcher pytorch


# #--------------------------------   SPARSE RCNN STUDENT   --------------------------------#

# MODEL_NAME='sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x'
# MODEL_NAME='sparse_rcnn_r50_fpn_mstrain_480-800_3x'
MODEL_NAME='sparse_rcnn_r101_fpn_mstrain_480-800_3x_fskd'
FOLDER_NAME='sparse_rcnn_kd'
CUDA_VISIBLE_DEVICES=0,4,5,6 python -m torch.distributed.launch \
                                --nproc_per_node=4 \
                                --master_port 1028 \
                                train.py \
                                --config configs/${FOLDER_NAME}/coco_${MODEL_NAME}.py \
                                --seed 0 \
                                --work-dir result/coco/debugging \
                                # --work-dir result/coco/${MODEL_NAME} \
                                --launcher pytorch
                                # --resume-from result/coco/${MODEL_NAME}/epoch_1.pth  \



