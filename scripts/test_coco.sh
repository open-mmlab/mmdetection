
FOLDER_NAMES=('mask_rcnn_kd' 'mask_rcnn_kd'  'sparse_rcnn_kd'  'sparse_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd'  'faster_rcnn_kd')
MODEL_NAMES=('mask_rcnn_r101_fpn_1x_fskd'  'mask_rcnn_r50_fpn_1x_fskd'  'sparse_rcnn_r101_fpn_mstrain_480-800_3x_fskd'  'sparse_rcnn_r50_fpn_mstrain_480-800_3x_fskd'  'faster_rcnn_r50_fpn_3x_fskd'  'faster_rcnn_r101_fpn_1x_fskd'  'faster_rcnn_r101_fpn_2x_fskd'  'faster_rcnn_r101_fpn_3x_fskd'  'faster_rcnn_x101_32x4d_fpn_1x_fskd'  'faster_rcnn_x101_32x4d_fpn_2x_fskd'  'faster_rcnn_x101_32x4d_fpn_3x_fskd'  'faster_rcnn_x101_64x4d_fpn_1x_fskd'  'faster_rcnn_x101_64x4d_fpn_2x_fskd'  'faster_rcnn_x101_64x4d_fpn_3x_fskd'  'faster_rcnn_r50_dc5_1x_fskd'  'faster_rcnn_r50_fpn_1x_fskd'  'faster_rcnn_r50_fpn_2x_fskd')
EPOCH_NUMBERS=('12'  '12'  '36'  '36'  '36'  '12'  '24'  '36'  '12'  '24'  '24'  '12'  '24'  '36'  '12'  '12'  '24')


for (( i = 0 ; i < ${#FOLDER_NAMES[@]} ; i++ )) ; do
    FOLDER_NAME=${FOLDER_NAMES[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    EPOCH_NUMBER=${EPOCH_NUMBERS[$i]}

    CONFIG_FILE='configs/'${FOLDER_NAME}'/coco_'${MODEL_NAME}'.py'
    CHECKPOINT_FILE='/ailab_mat/checkpoints/sung/msdet/coco_student/'${MODEL_NAME}'/epoch_'${EPOCH_NUMBER}'.pth'

    # test with single gpu
    CUDA_VISIBLE_DEVICES=4 python test.py \
        ${CONFIG_FILE} \
        ${CHECKPOINT_FILE} \
        --format-only \
        --cfg-options data.test.ann_file=/SSDb/sung/dataset/coco/annotations/image_info_test-dev2017.json data.test.img_prefix=/SSDb/sung/dataset/coco/test2017 \
        --eval-options jsonfile_prefix=/ailab_mat/checkpoints/sung/msdet/test-dev/${MODEL_NAME}
done



# # test with four gpus
# CUDA_VISIBLE_DEVICES=0,1,3,4 bash tools/dist_test.sh \
#                                 ${CONFIG_FILE} \
#                                 ${CHECKPOINT_FILE} \
#                                 4 \ # four gpus
#                                 --format-only \
#                                 --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 \
#                                 --eval-options jsonfile_prefix=${WORK_DIR}/results

