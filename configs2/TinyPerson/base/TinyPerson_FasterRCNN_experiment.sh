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


# add by lzj 用这个脚本可以生成对应的评价Json文件，生成的文件和--work-dir没关系，固定在./exp/latest_result.json里，因为在cocofmt.py里是在临时文件夹生成过后，直接按照固定路径copy过去的
tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/latest.pth \
      4 --work-dir exp/fasterrcnn_r50_fpn_lr0.02_1x_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640/old640x512_lr0.02_2x_4g/best_bbox_mAP_epoch_18.pth \
      4 --work-dir exp/faster_rcnn_seCBAM50_fpn_2x_TinyPerson640_lr0.02_1x_4g --eval bbox

# add by lzj 导出模型为onnx查看网络结构，发现CBAM模块被移动到最前面去了，怀疑是这方面的原因，但是不知道怎么解决这方面的问题
python tools/deployment/pytorch2onnx.py \
    configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
    ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
    --output-file ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 640 512 \
    --dynamic-export


