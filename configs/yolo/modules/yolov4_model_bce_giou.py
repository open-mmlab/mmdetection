# model settings
model = dict(
    type='YOLOV4',
    # TODO: fix pretrained model
    # pretrained='../checkpoints/yolov4/yolov4.conv.137.pth',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        csp_on=True,
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04, momentum=0.03),
    ),
    neck=dict(
        type='YOLOV4Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128],
        spp_on=True,
        spp_pooler_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04, momentum=0.03),
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(142, 110), (192, 243), (459, 401)],
                        [(36, 75), (76, 55), (72, 146)],
                        [(12, 16), (19, 36), (40, 28)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        norm_cfg=dict(type='BN', requires_grad=True, eps=1e-04, momentum=0.03),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0, reduction='sum')))
