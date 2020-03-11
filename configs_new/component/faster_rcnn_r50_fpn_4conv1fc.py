_base_ = './faster_rcnn_r50_fpn.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='ConvFCBBoxHead',
        num_shared_convs=4,
        num_shared_fcs=1,
        in_channels=256,
        conv_out_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=81,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
