_base_ = '../hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'

custom_hooks = [
    dict(
        type='RFSearchHook',
        mode='search',
        rfstructure_file=None,
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
