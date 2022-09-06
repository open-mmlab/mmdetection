_base_ = './cascade-mask-rcnn_r101_fpn_seesaw-loss_sample1e-3-ms-2x_lvis-v1.py'  # noqa: E501
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
