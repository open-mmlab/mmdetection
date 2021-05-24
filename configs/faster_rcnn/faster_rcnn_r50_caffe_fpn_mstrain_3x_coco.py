_base_ = 'faster_rcnn_r50_fpn_mstrain_3x_coco.py'
model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        norm_cfg=dict(requires_grad=False), norm_eval=True, style='caffe'))
