# 论文里的学习率是0.002,但是没有他官方放出来的config文件，还是没办法确定他用了什么trick能把点刷那么高的
# 论文里提到了他用了OHEM，有可能是通过这个刷的点。
# APtiny50 0.4670
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_sspnet_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

#2张卡位0.002,根据线性缩放原则，4张卡x2
export GPU=4 && LR=0.004 && CONFIG="faster_rcnn_r50_sspnet_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# 加入ohem尝试复现论文结果看看？
# APTiny50 0.5011,为什么还不如普通的？
# ApTiny50 0.4917
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_sspnet_ohem_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/ohem/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/ohem/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

#ApTiny50 0.4817
export GPU=4 && LR=0.004 && CONFIG="faster_rcnn_r50_sspnet_ohem_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/ohem/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/ohem/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}