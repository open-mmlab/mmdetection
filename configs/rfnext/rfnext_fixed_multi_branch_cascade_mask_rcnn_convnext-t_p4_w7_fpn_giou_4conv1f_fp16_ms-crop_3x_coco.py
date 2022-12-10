_base_ = '../convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py'  # noqa

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='fixed_multi_branch',
        rfstructure_file=  # noqa
        './configs/rfnext/search_log/convnext_cascade_maskrcnn/local_search_config_step11.json',  # noqa
        verbose=True,
        by_epoch=True,
        config=dict(
            search=dict(
                step=0,
                max_step=12,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                normlize='absavg',
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=[])))
]
