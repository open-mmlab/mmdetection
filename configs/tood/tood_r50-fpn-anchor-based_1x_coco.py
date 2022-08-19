_base_ = './tood_r50-fpn_1x_coco.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
