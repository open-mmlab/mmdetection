# epoch12 aptiny50 0.5758 tiny1 0.4345 tiny2 0.6116 tiny3 0.6675 small 0.7372
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_ssm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/fpn_lka_ssm/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/fpn_lka_ssm/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4