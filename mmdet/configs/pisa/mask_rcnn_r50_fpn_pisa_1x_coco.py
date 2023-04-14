if '_base_':
    from ..mask_rcnn.mask_rcnn_r50_fpn_1x_coco import *
from mmdet.models.roi_heads.pisa_roi_head import PISARoIHead
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.task_modules.samplers.score_hlr_sampler import ScoreHLRSampler

model.merge(
    dict(
        roi_head=dict(
            type=PISARoIHead,
            bbox_head=dict(
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0))),
        train_cfg=dict(
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                sampler=dict(
                    type=ScoreHLRSampler,
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                    k=0.5,
                    bias=0.),
                isr=dict(k=2, bias=0),
                carl=dict(k=1, bias=0.2))),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0))))
