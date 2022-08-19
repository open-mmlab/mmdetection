_base_ = './mask-rcnn_r50-fpn-random-seesaw-loss_mstrain-2x_lvis-v1.py'
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
