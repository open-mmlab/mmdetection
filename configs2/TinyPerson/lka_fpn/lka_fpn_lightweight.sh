# 直接改64失败，试试看将fc和randomsampler的设定也除以4看看
# loss一直降不下来，可能还是有问题，是不是resnet18就默认有问题
# 采用原本的train_cfg，loss正常下降，但是map还是很低
# 将bbox_head的FC_channels从1024降为256，结果还是一样，怀疑是不是有问题
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r18_lka_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4