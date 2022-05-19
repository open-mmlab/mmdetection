export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# epoch12 0.5721 0.4065 0.6117 0.6720 0.7310
# 当初就不知道哪个epoch训出来的……这下怎么办
# epoch10 0.5802 0.4140 0.6287 0.6849 0.7275
# epoch9 0.5661 0.4057 0.6033 0.6697
# epoch11 0.5662 0.4155 0.6084 0.7248
export GPU=2 && CONFIG="faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData" && CUDA_VISIBLE_DEVICES=0,1 PORT=10012 \
   tools/dist_test.sh configs2/TinyPerson/PAFPN/${CONFIG}.py \
   ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr0.02_1x_4g/epoch_11.pth \
   ${GPU} --work-dir exp/${CONFIG} --eval bbox

export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=2 && CONFIG="faster_rcnn_r50_pafpn_1x_TinyPerson640_newData" && CUDA_VISIBLE_DEVICES=2,3 PORT=10011 \
   tools/dist_test.sh configs2/TinyPerson/PAFPN/${CONFIG}.py \
   ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr0.02_1x_4g/epoch_9.pth \
   ${GPU} --work-dir exp/${CONFIG} --eval bbox