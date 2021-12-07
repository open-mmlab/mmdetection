export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_sewmp50_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_sewmp50_fpn_2x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# add by lzj exp1.3: Faster-FPN, 4 TESLA T4
export GPU=4 && LR=0.08 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10000 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# add by lzj Faster-FPN ,4 TESLA T4 大batch_size的结果还不如原本的ResNet50的结果，0.4462。采用batch-size为1的情况下试试
# 结果上还是不如原本的权重。这次试试把fronze_stage去掉试试。
# 去了frozen_stage，点数从0.44掉到了0.43，也不是这方面的问题。找了CBAM的权重，在通道方面的编码应该是一样的，拿来试试看
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_sewmp50_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_sewmp50_fpn_1x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 找到了CBAM在ImageNet上的权重，拿来试试看
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10002 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_sespatial50_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_sespatial50_fpn_2x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10003 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.08 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10003 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# add by lzj 实验证明还是原本的配置才是最优的结果，保持batch-size为1就别动了。复现结果为0.4987，根原本的0.4981差不多了，可以专心搞注意力了。
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.08 && EPOCH=3x && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10003 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640/old640x512_lr${LR}_${EPOCH}_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.02 && EPOCH=3x && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10003 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640/old640x512_lr${LR}_${EPOCH}_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.01 && EPOCH=300e && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10003 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640/old640x512_lr${LR}_${EPOCH}_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# add by lzj 试试carafe的效果 有效，APtiny 51.44
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 试试新的rfcr的效果 mAP 0.5021
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 将上采样模式改为bilinear，MBConv的进入通道改为与出通道相同都为96，试试mAP 0.4964
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640_MBConv96_96_bilinear/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 为了搞明白为什么掉点，尝试着把MBConv的进入通道数调回48，再看看结果 mAP 0.5038(+0.0051)
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640_MBConv48_96_bilinear/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 尝试看看没有MBConv的结果 mAP 0.4976,还不如baseline，虽然掉点不多，掉了0.0011
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640_noMBConv_bilinear/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 2x epoch的效果，跟第一次训练的结果不一样……不知道是哪的问题
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_2x_TinyPerson640_modifyMaxPooling/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 改用PAFPN试试，结果掉点了，APTiny50 0.4931
# PAFPN的情况上来看，第9轮的APTiny50比第12批高一点点，不知道是不是陷入到过拟合了……
# 把每一批次的都打出来做了一下测试，APTiny50 0.5008,比baseline高了0.0021，基本没用……
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 采用2x的schedule跑了一下看看，结果果然也没有1x的好,而且比原Baseline低
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10004 tools/dist_train.sh configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_2x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_2x_TinyPerson640/old640x512_lr${LR}_2x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# 修正了RFCR的maxpooling的kernel和stride的问题，重新跑个结果看看
# APtiny50 0.4992 基本没有提升
export GPU=4 && LR=0.02 && CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10008 tools/dist_train.sh configs2/TinyPerson/rfcr/faster_rcnn_r50_rfcr_fpn_1x_TinyPerson640.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/rfcr/faster_rcnn_r50_rfcr_bilinear_fpn_1x_TinyPerson640/old640x512_lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# ATSS相关尝试，mmdetection的官方issue中写的二阶段的网络没有办法只通过修改config文件就能使用ATSS的方法，需要修改代码，转而尝试在RetinaNet上进行ATSS的实验
# 网站上给出来baseline的就是这个文件，网站上的Ap_{tiny}^{50}=45.22，自己复现为0.4645
export GPU=4 && LR=0.02 && CONFIG="retinanet_r50_fpns4_1x_TinyPerson640_clipg"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/${CONFIG}/old640x512_lr${LR}_1x_clipg_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# atss复现的loss变为0，检测无结果。
export GPU=4 && LR=0.02 && CONFIG="retinanet_r50_atss_fpns4_1x_TinyPerson640_clipg"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/ATSS/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/ATSS/${CONFIG}/old640x512_lr${LR}_1x_clipg_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# APTiny50 0.4889
export GPU=2 && LR=0.01 && CONFIG="faster_rcnn_r50_sspnet_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# 第12epoch是0.5032，看看第10个epoch是怎么样的
# 第10epoch是0.4877
# 论文里提到是训练了10个epoch，但是采用了OHEM，不用的话效果还是不行
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_sspnet_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# 试一下2x的效果
# APTiny50=0.4894，还是默认的1x效果较好
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_sspnet_2x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# 论文里的学习率是0.002,但是没有他官方放出来的config文件，还是没办法确定他用了什么trick能把点刷那么高的
# 论文里提到了他用了OHEM，有可能是通过这个刷的点。
# APtiny50 0.4670
export GPU=4 && LR=0.002 && CONFIG="faster_rcnn_r50_sspnet_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

# 加入ohem尝试复现论文结果看看？
# APTiny50 0.5011,为什么还不如普通的？
export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_sspnet_ohem_1x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/ohem/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/ohem/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}

export GPU=4 && LR=0.02 && CONFIG="faster_rcnn_r50_sspnet_ohem_2x_TinyPerson640"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/sspnet/ohem/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/ohem/${CONFIG}/old640x512_lr${LR}_2x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR}







