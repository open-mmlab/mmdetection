_base_ = '../panoptic_fpn/panoptic_fpn_r2_50_fpn_fp16_1x_coco.py'

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='fixed_single_branch',
        rfstructure_file=  # noqa
        './configs/rfnext/search_log/panoptic_fpn_r2_50_fpn_fp16_1x_coco/local_search_config_step10.json',  # noqa
        verbose=True,
        by_epoch=True,
        config=dict(
            search=dict(
                step=0,
                max_step=11,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=['stem', 'layer1'])))
]
