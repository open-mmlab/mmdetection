_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
classes = ('person', )
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'  # noqa
