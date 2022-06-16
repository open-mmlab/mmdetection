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
        logdir='./search_log/convnext_cascade_maskrcnn',
        mode='search',
        rfstructure_file=None,
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
                S=2,
                finetune=False,
                skip_layer=['stem', 'layer1'])),
    ))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='RFSearch',
        logdir=model['rfsearch_cfg']['logdir'],
        config=model['rfsearch_cfg']['config'],
        mode=model['rfsearch_cfg']['mode'],
    ),
]
