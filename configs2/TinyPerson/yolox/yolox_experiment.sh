
export GPU=4 && LR=0.005 && CONFIG="yolox_l_8x8_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/yolox/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/yolox/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4