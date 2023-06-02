_base_ = './reppoints-moment_r50_fpn-gn_head-gn_1x_coco.py'
model = dict(bbox_head=dict(transform_method='partial_minmax'))
