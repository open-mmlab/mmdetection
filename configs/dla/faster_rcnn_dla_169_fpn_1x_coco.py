_base_ = './faster_rcnn_dla_60_fpn_1x_coco.py'

model = dict(pretrained='open-mmlab://dla169', backbone=dict(depth=169))
