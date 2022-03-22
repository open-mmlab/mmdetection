# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .kd_one_stage import KnowledgeDistillationSingleStageDetector


@DETECTORS.register_module()
class LAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_ckpt,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KnowledgeDistillationSingleStageDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)
        self.eval_teacher = eval_teacher
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = build_head(teacher_bbox_head)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # get label assignment from the teacher
        with torch.no_grad():
            x_teacher = self.extract_teacher_feat(img)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            label_assignment_results = \
                self.teacher_model.bbox_head.get_label_assignment(
                    *outs_teacher, gt_bboxes, gt_labels, img_metas,
                    gt_bboxes_ignore)

        # the student use the label assignment from the teacher to learn
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, label_assignment_results,
                                              img_metas, gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses
