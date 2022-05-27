# add by lzj 用这个脚本可以生成对应的评价Json文件，生成的文件和--work-dir没关系，固定在./exp/latest_result.json里，因为在cocofmt.py里是在临时文件夹生成过后，直接按照固定路径copy过去的
# 相关文件已删除，仅作示范用
tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/latest.pth \
      4 --work-dir exp/fasterrcnn_r50_fpn_lr0.02_1x_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/latest.pth \
      1 --work-dir exp/fasterrcnn_r50_fpn_lr0.02_1x_1g --eval bbox

export LR=0.01 && EPOCH=300e tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_seCBAM50_fpn_${EPOCH}_TinyPerson640/old640x512_lr${LR}_${EPOCH}_4g/best_bbox_mAP_epoch_45.pth \
      4 --work-dir exp/faster_rcnn_seCBAM50_fpn_TinyPerson640_lr${LR}_${EPOCH}_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
      4 --work-dir exp/faster_rcnn_seCBAM50_fpn_TinyPerson640_lr0.01_1x_4g --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.01_1x_4g/test_result
# show-dir和show只支持单GPU，应该有别的办法
python tools/test.py configs2/TinyPerson/base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
    --work-dir exp/faster_rcnn_r50_carafe_fpn_TinyPerson640_lr0.01_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.01_1x_4g/test_result

# 查看carafe的结果
tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
      1 --work-dir exp/fasterrcnn_r50_carafe_fpn_lr0.02_1x_1g --eval bbox

# 查看carafe的结果_新数据
# 0.5850
tools/dist_test.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
      4 --work-dir exp/fasterrcnn_r50_carafe_fpn_lr0.02_1x_1g --eval bbox

tools/dist_test.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_pytorch_fpn_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
      4 --work-dir exp/fasterrcnn_r50_carafe_pytorch_fpn_lr0.02_1x_1g --eval bbox

# 0.5682
tools/dist_test.sh configs2/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/carafe/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_carafe_fpn_lr0.02_1x_1g --eval bbox

#sspnet的结果
tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640/old640x512_lr0.02_1x_4g/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/model_zoo/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

# 查看pafpn_unified_carafe的结果，顺便看看 保存的图片使什么样的
python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result \

python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result_nolabel_fuseconvbn \
    --show-score-thr 0.7 --fuse-conv-bn

python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result_nolabel_fuseconvbn_nosubimg \
    --show-score-thr 0.7 --fuse-conv-bn

# 查看转换后的文件是否对得上
python  tools/misc/browse_dataset.py   configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData.py
tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_coco.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/model_zoo/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

# 查看pafpn_unified_carafe的结果，顺便看看 保存的图片使什么样的
python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result \

python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result_nolabel_fuseconvbn \
    --show-score-thr 0.7 --fuse-conv-bn

python tools/test.py configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_10.pth \
    --work-dir exp/faster_rcnn_r50_pafpn_unified_carafe_TinyPerson640_newData_lr0.02_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_unified_carafe_1x_TinyPerson640_newData/old640x512_lr0.02_1x_4g/test_result_nolabel_fuseconvbn_nosubimg \
    --show-score-thr 0.7 --fuse-conv-bn

# 查看转换后的文件是否对得上
python  tools/misc/browse_dataset.py   configs2/TinyPerson/PAFPN/faster_rcnn_r50_pafpn_attetion_carafe_1x_TinyPerson640_newData.py

# APTiny50 0.5579
# epoch12 aptiny50 0.5573 tiny1 0.4055 tiny2 0.5979 tiny3 0.6553 small 0.7166
export GPU=4 && LR=0.016 && CONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640_newData"&& CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 \
  tools/dist_train.sh configs2/TinyPerson/base/${CONFIG}.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/base/${CONFIG}/old640x512_lr${LR}_1x_${GPU}g/  \
  --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8

export GPU=4 && CONFIG="faster_rcnn_r50_fpn_1x_TinyPerson640_newData"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=10001 python tools/test.py configs2/TinyPerson/base/${CONFIG}.py \
   ../TOV_mmdetection_cache/work_dir/TinyPerson/base/${CONFIG}/old640x512_lr0.016_1x_4g/best_bbox_mAP_epoch_12.pth \
  --work-dir exp/${CONFIG} \
  --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/base/${CONFIG}/old640x512_lr0.016_1x_4g/test_result_img/ \
  --show-score-thr 0.7
