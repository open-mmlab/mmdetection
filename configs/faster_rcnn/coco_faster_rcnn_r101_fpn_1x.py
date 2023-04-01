_base_ = './coco_faster_rcnn_r50_fpn_1x.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4
)