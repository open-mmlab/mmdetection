_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(type='GIoULoss', loss_weight=1.0))))
