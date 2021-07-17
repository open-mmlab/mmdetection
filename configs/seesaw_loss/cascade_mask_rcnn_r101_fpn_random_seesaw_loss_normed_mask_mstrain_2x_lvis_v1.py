_base_ = './cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py'  # noqa: E501
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
