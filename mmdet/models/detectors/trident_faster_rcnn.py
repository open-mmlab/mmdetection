import copy

import torch
from mmcv import ConfigDict

from .faster_rcnn import FasterRCNN

from .two_stage import TwoStageDetector
from ..builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
class TridentFasterRCNN(FasterRCNN):

    def __init__(self,
                 num_branch,
                 test_branch_idx,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):

        backbone['num_branch'] = num_branch
        backbone['test_branch_idx'] = test_branch_idx
        roi_head['num_branch'] = num_branch
        roi_head['test_branch_idx'] = test_branch_idx
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super(TridentFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )

    # def __init__(self, num_branch, **kwargs):
    #     self.num_branch=num_branch
    #     super(TridentFasterRCNN, self).__init__(**kwargs)
    #     # for (k, v) in self.named_parameters():
    #     #     print("%-60s "%(k), v.shape)

    # def simple_test_rpn(self, x, img_metas):
    #     assert len(img_metas) == 1
    #     num_branch = (self.num_branch
    #                   if self.test_branch_idx==-1 else 1)
    #     trident_img_metas = img_metas * num_branch
    #     rpn_score, rpn_bbox = self.rpn_head(x)
    #
    #     proposal_list = self.rpn_head.get_bboxes(
    #         rpn_score, rpn_bbox,
    #         trident_img_metas
    #     )
    #     return proposal_list

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        # import joblib
        # img =joblib.load('/home/nirvana/temp/images.pkl')
        # gt_bboxes = joblib.load('/home/nirvana/temp/gt_bboxes.pkl')
        # gt_labels = joblib.load('/home/nirvana/temp/gt_labels.pkl')
        # img_metas[0]['img_shape'] = (704, 1053)
        # img_metas[1]['img_shape'] = (736, 1108)
        # for i in range(2):
        #     img_metas[i]['pad_shape'] = img_metas[i]['img_shape']
        #     img_metas[i]['scale_factor'] = None
        #     img_metas[i]['ori_shape'] = None

        x = self.extract_feat(img)
        losses = dict()
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (
                trident_gt_bboxes, trident_img_metas)
            if gt_bboxes_ignore is not None:
                print(gt_bboxes_ignore)
                assert False
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(
                *rpn_outs, trident_img_metas, cfg=proposal_cfg)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, trident_img_metas,
            proposal_list,
            trident_gt_bboxes,
            trident_gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            **kwargs
        )
        losses.update(roi_losses)

        return losses
