export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/VisDrone/attention_carafe/${CONFIG}.py $GPU \
  --work-dir ../VisDrone_cache/attention_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1