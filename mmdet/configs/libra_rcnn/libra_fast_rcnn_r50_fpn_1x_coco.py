if '_base_':
    from ..fast_rcnn.fast_rcnn_r50_fpn_1x_coco import *
from mmdet.models.necks.fpn import FPN
from mmdet.models.necks.bfp import BFP
from mmdet.models.losses.balanced_l1_loss import BalancedL1Loss
from mmdet.models.task_modules.samplers.combined_sampler import CombinedSampler
from mmdet.models.task_modules.samplers.instance_balanced_pos_sampler import InstanceBalancedPosSampler
from mmdet.models.task_modules.samplers.iou_balanced_neg_sampler import IoUBalancedNegSampler
# model settings
model.merge(
    dict(
        neck=[
            dict(
                type=FPN,
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            dict(
                type=BFP,
                in_channels=256,
                num_levels=5,
                refine_level=2,
                refine_type='non_local')
        ],
        roi_head=dict(
            bbox_head=dict(
                loss_bbox=dict(
                    _delete_=True,
                    type=BalancedL1Loss,
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rcnn=dict(
                sampler=dict(
                    _delete_=True,
                    type=CombinedSampler,
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type=InstanceBalancedPosSampler),
                    neg_sampler=dict(
                        type=IoUBalancedNegSampler,
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3))))))

# MMEngine support the following two ways, users can choose
# according to convenience
# train_dataloader.dataset.proposal_file = 'libra_proposals/rpn_r50_fpn_1x_train2017.pkl'  # noqa
train_dataloader.merge(
    dict(
        dataset=dict(
            proposal_file='libra_proposals/rpn_r50_fpn_1x_train2017.pkl')))

# val_dataloader.dataset.proposal_file = 'libra_proposals/rpn_r50_fpn_1x_val2017.pkl'  # noqa
# test_dataloader = val_dataloader
val_dataloader.merge(
    dict(
        dataset=dict(
            proposal_file='libra_proposals/rpn_r50_fpn_1x_val2017.pkl')))
test_dataloader = val_dataloader
