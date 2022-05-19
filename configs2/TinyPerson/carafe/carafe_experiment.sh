# newdata carafe
# epoch12 APTiny50 0.5850
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# pytorch复现版，https://github.com/XiaLiPKU/CARAFE
# epoch11 0.5819 epoch10 0.5816
# enc后面加CBAM epoch 0.5818
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}


export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_CBAM_fpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_2x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_CBAM_fpn_2x_TinyPerson640_newData/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_2x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_CBAM_fpn_2x_TinyPerson640_newData/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}


export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_16e_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_CBAM_fpn_16e_TinyPerson640_newData/old640x512_lr${LR}_16e_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} \
  --resume-from ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_CBAM_fpn_2x_TinyPerson640_newData/old640x512_lr${LR}_2x_${GPU}g/epoch_10.pth

# carafe_origin
# epoch12 ApTiny50 0.5813
export GPU=4 && LR=0.008 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}
