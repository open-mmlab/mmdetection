# epoch9 aptiny 0.5786 tiny1 0.4393 tiny2 0.6263 tiny3 6629 small 0.7344
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4


