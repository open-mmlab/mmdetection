_base_ = './mask_rcnn_r50_fpn_random_seesaw_loss_mstrain_2x_lvis_v1.py'
model = dict(
    roi_head=dict(
        mask_head=dict(
            predictor_cfg=dict(type='NormedConv2d', tempearture=20))))
