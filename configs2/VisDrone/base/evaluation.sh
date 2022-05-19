export GPU=4 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 python tools/test.py configs2/VisDrone/base/${CONFIG}.py \
   ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr0.04_1x_4g/best_bbox_mAP_epoch_12.pth \
  --work-dir exp/${CONFIG} \
  --show-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr0.04_1x_4g/test_result_img/ \
  --show-score-thr 0.7


export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8
