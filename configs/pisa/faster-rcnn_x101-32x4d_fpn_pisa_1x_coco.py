_base_ = '../faster_rcnn/faster-rcnn_x101-32x4d_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        type='PISARoIHead',
        bbox_head=dict(
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            sampler=dict(
                type='ScoreHLRSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                k=0.5,
                bias=0.),
            isr=dict(k=2, bias=0),
            carl=dict(k=1, bias=0.2))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))
