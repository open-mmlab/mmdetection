# best bbox_map_s 0.1470
# bbox_mAP: 0.2120, bbox_mAP_50: 0.4410, bbox_mAP_75: 0.1820, bbox_mAP_s: 0.1470, bbox_mAP_m: 0.3080, bbox_mAP_l: 0.2430,
# bbox_mAP_copypaste: 0.212 0.441 0.182 0.147 0.308 0.243
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8

# Epoch(val) [12][274]   bbox_mAP: 0.2630, bbox_mAP_50: 0.5210, bbox_mAP_75: 0.2360, bbox_mAP_s: 0.1760, bbox_mAP_m: 0.3790, bbox_mAP_l: 0.4220, bbox_mAP_copypaste: 0.263 0.521 0.236 0.176 0.379 0.422
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r101_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r101_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr0.02_1x_4g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4 \
  --resume ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr0.02_1x_4g/latest.pth


export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r101_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8

# 实验mosaic的效果，结果等于基本没有，不知道是哪的问题，换成YOLOX的train_pipeline试试
export GPU=4 && LR=0.005 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640_mosaic"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/origin_mosaic_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 采用COCO的文件设置，best_epoch12点数更低，ap_s 0.1300
export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r50_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/base/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/base/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8