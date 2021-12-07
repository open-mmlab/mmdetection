# add by lzj 导出模型为onnx查看网络结构，发现CBAM模块被移动到最前面去了，怀疑是这方面的原因，但是不知道怎么解决这方面的问题
python tools/deployment/pytorch2onnx.py \
    configs2/TinyPerson/base/faster_rcnn_r50_fpn_1x_TinyPerson640.py \
    ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.pth \
    --output-file ../TOV_mmdetection_cache/work_dir/TinyPerson/Base/faster_rcnn_r50_fpn_1x_TinyPerson640_baseline_0.4987mAP/old640x512_lr0.02_1x_4g/best_bbox_mAP_epoch_12.onnx \
    --input-img demo/demo.jpg \
    --test-img tests/data/color.jpg \
    --shape 640 512 \
    --dynamic-export