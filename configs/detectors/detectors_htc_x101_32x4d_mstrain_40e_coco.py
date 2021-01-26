_base_ = './detectors_htc_r50_mstrain_40e_coco.py'
model = dict(
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='DetectoRS_ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        #conv_cfg=dict(type='ConvAWS'),
        #sac=dict(type='SAC', use_deform=True),
        #stage_with_sac=(False, True, True, True),
        #output_img=True
    ),
    neck=dict(
        type='RFP',
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNeXt',
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='open-mmlab://resnext101_32x4d',
            style='pytorch')))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
