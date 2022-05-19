# Epoch(val) [12][274]   bbox_mAP: 0.2680, bbox_mAP_50: 0.5220, bbox_mAP_75: 0.2460, bbox_mAP_s: 0.1840, bbox_mAP_m: 0.3850, bbox_mAP_l: 0.4180,
# bbox_mAP_copypaste: 0.268 0.522 0.246 0.184 0.385 0.418
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1 PORT=10004 tools/dist_train.sh configs2/VisDrone/pafpn_lkaattention/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/pafpn_lkaattention/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# Epoch(val) [11][137]   bbox_mAP: 0.2650, bbox_mAP_50: 0.5140, bbox_mAP_75: 0.2430, bbox_mAP_s: 0.1830, bbox_mAP_m: 0.3750, bbox_mAP_l: 0.4270,
# bbox_mAP_copypaste: 0.265 0.514 0.243 0.183 0.375 0.427
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_pafpn_lka_noaem_noffm_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10014 tools/dist_train.sh configs2/VisDrone/pafpn_lkaattention/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/pafpn_lkaattention/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4