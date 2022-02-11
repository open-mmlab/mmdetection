_base_ = './faster_rcnn_r50_caffe_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        # FrozenBN can only be used for BN with requires_grad=False and
        # norm_eval=True, which can save GPU memory.
        norm_cfg=dict(_delete_=True, type='FrozenBN')))
