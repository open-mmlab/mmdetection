export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
tools/dist_train.sh configs2/VisDrone/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=16

export GPU=1 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_VisDrone640"
python tools/test.py configs2/VisDrone/lka_fpn/${CONFIG}.py \
     ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12.pth \
    --work-dir exp/${CONFIG}_lr${LR}_1x_${GPU}g \
    --show-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/test_result \
    --show-score-thr 0.7 --fuse-conv-bn