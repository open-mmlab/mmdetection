# 从第一个epoch的损失来看，是不是要把学习率调低点？比如0.01
# epoch11 aptiny 0.5442 tiny1 0.3970 tiny2 0.5972 tiny3 0.6418 small 0.6902
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 半路停掉了，感觉学习率设小了跑不上去 epoch5 aptiny 0.3562 tiny1 0.2636 tiny2 0.2636 tiny3 0.4359 small 0.4834
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2 \
  --resume-from ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/latest.pth

# epoch9 aptiny 0.5449 tiny1 0.4186 tiny2 0.6002 tiny3 0.6294 small 0.6612
export GPU=4 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 每一层的注意力不变动分辨率直接用上一层或下一层的，看起来是比原来的要强
# epoch12 aptiny 0.5574 tiny1 0.4167 tiny2 0.6015 tiny3 0.6492 samll 0.7041
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g_carafeAttentionModified/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# epoch18 aptiny 0.5720 tiny1 0.4123 tiny2 0.6241 tiny3 0.6787 small 0.7402 效果可以啊,总算口岸这不掉点了 呵呵哒
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_2x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_2x_${GPU}g_carafeAttentionModified/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2 \
  --resume-from ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr0.01_2x_2g_carafeAttentionModified/latest.pth

export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_2x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_2x_${GPU}g_no_pafpn_att/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# pafpn
# epoch12 aptiny50 0.5919 tiny1 0.4512 tiny2 0.6376 tiny3 0.6838 small 0.7404
# 这个可能需要重跑
# epoch12 aptiny50 0.5837 tiny1 0.4319 tiny2 0.6261 tiny3 0.6779 small 0.7346
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_unified_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention_unified_carafe/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 这个虽然训完了，但是看样也得重新跑，哎
# epoch12 aptiny50 0.5875 tiny1 0.4334 tiny2 0.6320 tiny3 0.6807 small 0.7500
export GPU=2 && LR=0.02 && CONFIG="faster_rcnn_r50_pafpn_lkaattention_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/pafpn_lkaattention/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/pafpn_lkattention_unified_carafe/${CONFIG}/old640x640_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4

