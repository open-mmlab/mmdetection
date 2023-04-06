_base_ = '../fast_rcnn/fast-rcnn_r50-caffe_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.5),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.65, neg_iou_thr=0.65, min_pos_iou=0.65),
            sampler=dict(num=256))),
    test_cfg=dict(rcnn=dict(score_thr=1e-3)))

# MMEngine support the following two ways, users can choose
# according to convenience
# train_dataloader = dict(dataset=dict(proposal_file='proposals/crpn_r50_caffe_fpn_1x_train2017.pkl'))  # noqa
_base_.train_dataloader.dataset.proposal_file = 'proposals/crpn_r50_caffe_fpn_1x_train2017.pkl'  # noqa

# val_dataloader = dict(dataset=dict(proposal_file='proposals/crpn_r50_caffe_fpn_1x_val2017.pkl'))  # noqa
# test_dataloader = val_dataloader
_base_.val_dataloader.dataset.proposal_file = 'proposals/crpn_r50_caffe_fpn_1x_val2017.pkl'  # noqa
test_dataloader = _base_.val_dataloader

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))
