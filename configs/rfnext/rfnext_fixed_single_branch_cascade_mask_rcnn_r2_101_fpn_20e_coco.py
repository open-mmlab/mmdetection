_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'
model = dict(
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    rfsearch_cfg=dict(
        mode='fixed_single_branch',
        rfstructure_file=  # noqa
        './configs/rfnext/search_log/cascade_mask_rcnn_r2_101_fpn_20e_coco/local_search_config_step11.json',  # noqa
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
                skip_layer=['stem', 'layer1'])),
    ))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='RFSearchHook',
        config=model['rfsearch_cfg']['config'],
        mode=model['rfsearch_cfg']['mode'],
    ),
]
