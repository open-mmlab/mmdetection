if '_base_':
    from ..fast_rcnn.fast_rcnn_r50_caffe_fpn_1x_coco import *
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        roi_head=dict(
            bbox_head=dict(
                bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
                loss_cls=dict(
                    type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.5),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rcnn=dict(
                assigner=dict(
                    pos_iou_thr=0.65, neg_iou_thr=0.65, min_pos_iou=0.65),
                sampler=dict(num=256))),
        test_cfg=dict(rcnn=dict(score_thr=1e-3))))

# MMEngine support the following two ways, users can choose
# according to convenience
# train_dataloader = dict(dataset=dict(proposal_file='proposals/crpn_r50_caffe_fpn_1x_train2017.pkl'))  # noqa
train_dataloader.dataset.proposal_file = 'proposals/crpn_r50_caffe_fpn_1x_train2017.pkl'  # noqa

# val_dataloader = dict(dataset=dict(proposal_file='proposals/crpn_r50_caffe_fpn_1x_val2017.pkl'))  # noqa
# test_dataloader = val_dataloader
val_dataloader.dataset.proposal_file = 'proposals/crpn_r50_caffe_fpn_1x_val2017.pkl'  # noqa
test_dataloader = val_dataloader

optim_wrapper.merge(dict(clip_grad=dict(max_norm=35, norm_type=2)))
