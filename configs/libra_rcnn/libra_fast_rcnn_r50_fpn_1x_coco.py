_base_ = '../fast_rcnn/fast_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    rcnn=dict(
        sampler=dict(
            _delete_=True,
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3))))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    train=dict(proposal_file=data_root +
               'libra_proposals/rpn_r50_fpn_1x_train2017.pkl'),
    val=dict(proposal_file=data_root +
             'libra_proposals/rpn_r50_fpn_1x_val2017.pkl'),
    test=dict(proposal_file=data_root +
              'libra_proposals/rpn_r50_fpn_1x_val2017.pkl'))
