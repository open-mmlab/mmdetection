# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .faster_rcnn_r50_caffe_fpn_ms_1x_coco import *

model.update(dict(roi_head=dict(bbox_head=dict(num_classes=1))))

metainfo = {
    'classes': ('person', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_dataloader.update(dict(dataset=dict(metainfo=metainfo)))
val_dataloader.update(dict(dataset=dict(metainfo=metainfo)))
test_dataloader.update(dict(dataset=dict(metainfo=metainfo)))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'  # noqa
