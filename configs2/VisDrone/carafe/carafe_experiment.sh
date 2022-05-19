# Epoch(val) [12][274]   bbox_mAP: 0.2690, bbox_mAP_50: 0.5290, bbox_mAP_75: 0.2470, bbox_mAP_s: 0.1870, bbox_mAP_m: 0.3830, bbox_mAP_l: 0.4320,
# bbox_mAP_copypaste: 0.269 0.529 0.247 0.187 0.383 0.432
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_fpn_carafe_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1 PORT=10004 tools/dist_train.sh configs2/VisDrone/carafe/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/carafe/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2