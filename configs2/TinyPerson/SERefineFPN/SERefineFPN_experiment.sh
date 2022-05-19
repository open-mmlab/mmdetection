# APTiny50 0.5579
export GPU=4 && LR=0.016 && CONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640_newData"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.004 && CONFIG="faster_rcnn_r50_SERefineFPN_1x_TinyPerson640_newData"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/SERefineFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/SERefineFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR} \
  --resume-from ../TOV_mmdetection_cache/work_dir/TinyPerson/SERefineFPN/${CONFIG}/old640x512_lr0.016_1x_${GPU}g/latest.pth

export GPU=4 && LR=0.008 && CONFIG="faster_rcnn_r50_SERefineFPN_1x_TinyPerson640_newData"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/SERefineFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/SERefineFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}