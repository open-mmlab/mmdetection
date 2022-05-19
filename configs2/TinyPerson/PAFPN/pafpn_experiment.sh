# 新数据下的PAFPN测试
# epoch9 0.5722 看来之前的PAFPN效果不好是因为数据的问题
# epoch9 0.5722 0.4107 0.6175 0.6770 0.7375
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# backbone换101试试看？效果不知道会不会好
# epoch10 aptiny 0.5634 tiny1 0.4123 tiny2 0.4123 tiny3 0.6028 tiny4 0.6700 small 0.7340
export GPU=4 && LR=0.02 && CONFIG='faster_rcnn_r101_pafpn_1x_TinyPerson640_newData'
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4


# 0.5798,tensorboard波形不对，降一下学习率看看
export GPU=4 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 0.5867,从第3个epoch开始就到了0.52,然后就波动，怀疑学习率还是高，再调一下看看？
# 把padding_mode从reflect调回默认的看看
# 0.5804,还确实不如
# padding_mode 改成了replicate 之后变成了0.5800,还得是改回去把
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 0.5867，具体的情况因为崩了没出来
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_padding_reflect_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# loss爆炸了，换低点的学习率
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 重新跑一下吧，4卡的那种，另外，怀疑是最后设计的过程中通道注意力最后是2*sigmoid的缘故
# 到第9个epoch还是0.27左右，太次了，不跑了直接把2*去掉
# 去掉了之后还是那个德行
# 代码中把SE模块变成原本的实现，验证集上的AP一直上不去，到0.4513之后就不涨了
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1 \
  --resume-from ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr0.001_1x_2g/latest.pth

# 结果很垃圾，0.3133，再跑最后一个剩下的就跑对照试验了
export GPU=4 && LR=0.001 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1

# epoch8 aptiny 0.4983 tiny1 0.4663 tiny2 0.6895 tiny3 0.7356 tiny4 0.7085
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1

# 换为了CBAM中的spatial_gate试试看
# 它不work啊我的妈呀……咋整啊……
# epoch7 aptiny 0.3602 tiny1 0.3117 tiny2 0.4628 tiny3 0.3860 small 0.3786
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/attention_PAFPN/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=1



# 0.55多，不如0.02的版本
export GPU=4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 采用SSPNet的sampler
# 0.5835 epoch12 tensorboard波形不对，降一下学习率看看
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

#0.5724 节奏不太对
export GPU=4 && LR=0.008 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=4 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_icnegsampler_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

# 从tensorboard打出来的波形来看，有可能是参数的问题,感觉像学习率过大，再试试看
# 收敛的比标准的1x的schedule要快，在5epoch就达到了最好，
# epoch5 0.5268 epoch11 0.5368 还是不如原本的好
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# epoch10 0.5731 改进不多？那该咋办？
export GPU=4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 在FPN前面加了模块，效果不如原来好，查看曲线怀疑学习率的问题，降低学习率重新训练看看
# 把samples_per_gpu变到了4，看看炸不炸显存把
export GPU=4 && LR=0.004 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=2

export GPU=4 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_FPNRefine_1x_TinyPerson640_newData/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4