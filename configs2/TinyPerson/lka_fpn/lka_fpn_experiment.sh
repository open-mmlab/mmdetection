# epoch12 aptiny 0.5509 还是没有baseline好……难不成还是要用SSPNet的方式吗
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny 0.5739 tiny1 0.4277 tiny2 0.6104 tiny3 0.6702 small 0.7286
# epoch10 aptiny 0.5734 tiny1 0.4261 tiny2 0.6100 tiny3 0.6650 small 0.7339
# 我曹加了BN和ReLU后居然真的有用！
# 原SSPNet：APTiny_50 0.5825 第9个epoch结果最好
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_withbnrelu/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# 确实更有用。epoch12 aptiny 0.5832 tiny1 0.4297 tiny2 0.6297 tiny3 0.6789 small 0.7371
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_withbnrelu_withsigmoid/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# 测试一下采用放大分辨率的注意力特征图的方法
# 没练完就发现了不得了的事情。居然比之前的值更高。看来pafpn的实验也得重做
# epoch11 aptiny50 0.5910 tiny1 0.4450 tiny2 0.6359 tiny3 0.6846 small 0.7472
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_withbnrelu_withsigmoid_withlastatt/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# 加入carafe试试
# epoch12 aptiny 0.5861 tiny1 0.4370 tiny2 0.6326 tiny3 0.6778 small 0.7431
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_carafe_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_withbnrelu_withsigmoid/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_lka_fpn_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_carafe_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# epoch12 aptiny50 0.5715 tiny1 0.4400 tiny2 0.6053 tiny3 0.6528 small 0.7268
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noffm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noffm/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny50 0.5798 tiny1 0.4340 tiny2 0.6135 tiny3 0.6735 small 0.7286
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noaem_noffm/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny50 0.5789 tiny1 0.4447 tiny2 0.6156 tiny3 0.6650 small 0.7304
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noffm/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny50 0.5912 tiny1 0.4402 tiny2 0.6345 tiny3 0.6801 small 0.7429
# epoch11 aptiny50 0.5935 tiny1 0.4441 tiny2 0.6399 tiny3 0.6843 small 0.7449
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_1x_aem_ffm_attkernelsize7_attkerneldial2_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# 原精度 # epoch12 aptiny50 0.5798 tiny1 0.4340 tiny2 0.6135 tiny3 0.6735 small 0.7286
# epoch11 aptiny50 0.5816 tiny1 0.4421 tiny2 0.6175 tiny3 0.6707 small 0.7181
# epoch12 aptiny50 0.5784 tiny1 0.4355 tiny2 0.6122 tiny3 0.6700 small 0.7234
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_1x_attkernelsize7_attkerneldial2_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch11 aptiny50 0.5844 tiny1 0.4460 tiny2 0.6296 tiny3 0.6699 small 0.7216
# epoch12 aptiny50 0.5725 tiny1 0.4381 tiny2 0.6109 tiny3 0.6544 small 0.7256
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_1x_attkernelsize7_attkerneldial1_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny 0.5694 tiny1 0.4394 tiny2 0.6050 tiny3 0.6566 small 0.7234
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_nobnrelu_noconvsigmoid/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

# epoch12 aptiny50 0.5769 tiny1 0.4344 tiny2 0.6232 tiny3 0.6644 small 0.7292
export GPU=4 && LR=0.04 && CONFIG="faster_rcnn_r50_lka_fpn_noaem_noffm_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/lka_fpn/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/lka_fpn/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_noconvsigmoid/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4




