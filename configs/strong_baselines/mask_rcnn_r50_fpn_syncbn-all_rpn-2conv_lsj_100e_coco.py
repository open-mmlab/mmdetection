_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../common/lsj_100e_coco_instance.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
# Requires MMCV-full after  https://github.com/open-mmlab/mmcv/pull/1205.
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    # the model is trained from scratch, so init_cfg is None
    backbone=dict(
        frozen_stages=-1, norm_eval=False, norm_cfg=norm_cfg, init_cfg=None),
    neck=dict(norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),  # leads to 0.1+ mAP
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=head_norm_cfg),
        mask_head=dict(norm_cfg=head_norm_cfg)))
