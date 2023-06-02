_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    # 270k iterations with batch_size 64 is roughly equivalent to 144 epochs
    '../common/ssj_270k_coco-instance.py',
]

image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
# Use MMSyncBN that handles empty tensor in head. It can be changed to
# SyncBN after https://github.com/pytorch/pytorch/issues/36530 is fixed
head_norm_cfg = dict(type='MMSyncBN', requires_grad=True)
model = dict(
    # the model is trained from scratch, so init_cfg is None
    data_preprocessor=dict(
        # pad_size_divisor=32 is unnecessary in training but necessary
        # in testing.
        pad_size_divisor=32,
        batch_augments=batch_augments),
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
