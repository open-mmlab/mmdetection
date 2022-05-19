# aps epoch12  0.1750
# Epoch(val) [12][137]   bbox_mAP: 0.2590, bbox_mAP_50: 0.5120, bbox_mAP_75: 0.2320, bbox_mAP_s: 0.1750, bbox_mAP_m: 0.3710, bbox_mAP_l: 0.4180
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_pafpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/PAFPN/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8

export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_pafpn_2x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/PAFPN/${CONFIG}/slice_640x640_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8

# Epoch(val) [24][137]   bbox_mAP: 0.2770, bbox_mAP_50: 0.5370, bbox_mAP_75: 0.2580, bbox_mAP_s: 0.1890, bbox_mAP_m: 0.3990, bbox_mAP_l: 0.4390, bbox_mAP_copypaste: 0.277 0.537 0.258 0.189 0.399 0.439
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r101_pafpn_2x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/PAFPN/${CONFIG}/slice_640x640_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4


# aps epoch12 0.1800
# Epoch(val) [12][137]   bbox_mAP: 0.2630, bbox_mAP_50: 0.5170, bbox_mAP_75: 0.2390, bbox_mAP_s: 0.1800, bbox_mAP_m: 0.3760, bbox_mAP_l: 0.4210,
# bbox_mAP_copypaste: 0.263 0.517 0.239 0.180 0.376 0.421
export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_unified_carafe_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/PAFPN/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2