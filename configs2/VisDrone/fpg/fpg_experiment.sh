export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r50_fpg_50e_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/fpg/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/fpg/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r101_fpg_50e_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/fpg/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/fpg/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# Epoch(val) [24][137] bbox_mAP: 0.2670, bbox_mAP_50: 0.4810, bbox_mAP_75: 0.2650, bbox_mAP_s: 0.1810, bbox_mAP_m: 0.3900, bbox_mAP_l: 0.3970
export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r101_fpg_2x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/fpg/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/fpg/${CONFIG}/slice_640x640_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r101_fpg_50e_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1 PORT=10004 tools/dist_train.sh configs2/VisDrone/fpg/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/fpg/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2