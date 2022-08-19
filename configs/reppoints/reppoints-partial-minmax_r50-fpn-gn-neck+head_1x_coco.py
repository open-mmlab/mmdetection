_base_ = './reppoints-moment_r50-fpn-gn-neck+head_1x_coco.py'
model = dict(bbox_head=dict(transform_method='partial_minmax'))
