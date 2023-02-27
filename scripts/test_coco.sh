
MODEL_NAME='sparse_rcnn_r101_fpn_mstrain_480-800_3x'
CONFIG_FILE='configs/sparse_rcnn/coco_'${MODEL_NAME}'.py'
CHECKPOINT_FILE='result/coco/'${MODEL_NAME}'/epoch_36.pth'
WORK_DIR='/SSDb/heeseon/src/mmdetection'


# test with single gpu
CUDA_VISIBLE_DEVICES=4 python test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --format-only \
    --cfg-options data.test.ann_file=/SSDb/sung/dataset/coco/annotations/image_info_test-dev2017.json data.test.img_prefix=/SSDb/sung/dataset/coco/test2017 \
    --eval-options jsonfile_prefix=result/coco/${MODEL_NAME}
    # --eval-options jsonfile_prefix=${WORK_DIR}/results


# # test with four gpus
# CUDA_VISIBLE_DEVICES=0,1,3,4 bash tools/dist_test.sh \
#                                 ${CONFIG_FILE} \
#                                 ${CHECKPOINT_FILE} \
#                                 4 \ # four gpus
#                                 --format-only \
#                                 --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
#                                 --eval-options jsonfile_prefix=${WORK_DIR}/results

