_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
classes = ('person', 'bicycle', 'car')
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
# TODO: Update model url after bumping to V2.0
load_from = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'  # noqa
