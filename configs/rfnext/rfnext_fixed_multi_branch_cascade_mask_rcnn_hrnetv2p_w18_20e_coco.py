_base_ = '../hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'

custom_hooks = [
    dict(
        mode='fixed_multi_branch',
        rfstructure_file=  # noqa
        './configs/rfnext/search_log/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/local_search_config_step11.json',  # noqa
        verbose=True,
        by_epoch=True,
        config=dict(
            search=dict(
                step=0,
                max_step=12,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=[])))
]
