_base_ = './cascade-mask-rcnn_r101-fpn-random-seesaw-loss_mstrain-2x_lvis-v1.py'  # noqa: E501
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
