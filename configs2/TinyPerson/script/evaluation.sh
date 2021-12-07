# add by lzj 用这个脚本可以生成对应的评价Json文件，生成的文件和--work-dir没关系，固定在./exp/latest_result.json里，因为在cocofmt.py里是在临时文件夹生成过后，直接按照固定路径copy过去的
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
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
    --work-dir exp/faster_rcnn_r50_carafe_fpn_TinyPerson640_lr0.01_1x_4g \
    --show-dir ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.01_1x_4g/test_result

# 查看carafe的结果
tools/dist_test.sh configs2/TinyPerson/base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_carafe_fpn_1x_TinyPerson640/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
      1 --work-dir exp/fasterrcnn_r50_carafe_fpn_lr0.02_1x_1g --eval bbox

#sspnet的结果
tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640/old640x512_lr0.02_1x_4g/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_TinyPerson640.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/model_zoo/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

tools/dist_test.sh configs2/TinyPerson/sspnet/faster_rcnn_r50_sspnet_1x_coco.py \
     ../TOV_mmdetection_cache/work_dir/TinyPerson/sspnet/model_zoo/epoch_10.pth \
      4 --work-dir exp/fasterrcnn_r50_sspnet_lr0.02_1x_4g --eval bbox

