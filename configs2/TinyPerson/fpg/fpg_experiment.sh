#epoch20 aptiny50 0.5458 tiny1 0.3945 tiny2 0.5910 tiny3 0.6401 small 0.7042
export GPU=4 && LR=0.01 && CONFIG='faster_rcnn_r101_fpg_50e_TinyPerson640_newData'
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/fpg/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/fpg/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2