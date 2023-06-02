# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType
from ..utils.misc import unpack_gt_instances
from .kd_one_stage import KnowledgeDistillationSingleStageDetector


@MODELS.register_module()
class LAD(KnowledgeDistillationSingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 teacher_backbone: ConfigType,
                 teacher_neck: ConfigType,
                 teacher_bbox_head: ConfigType,
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        super(KnowledgeDistillationSingleStageDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)
        self.eval_teacher = eval_teacher
        self.teacher_model = nn.Module()
        self.teacher_model.backbone = MODELS.build(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_model.neck = MODELS.build(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_model.bbox_head = MODELS.build(teacher_bbox_head)
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

    @property
    def with_teacher_neck(self) -> bool:
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self.teacher_model, 'neck') and \
            self.teacher_model.neck is not None

    def extract_teacher_feat(self, batch_inputs: Tensor) -> Tensor:
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_model.backbone(batch_inputs)
        if self.with_teacher_neck:
            x = self.teacher_model.neck(x)
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        # get label assignment from the teacher
        with torch.no_grad():
            x_teacher = self.extract_teacher_feat(batch_inputs)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            label_assignment_results = \
                self.teacher_model.bbox_head.get_label_assignment(
                    *outs_teacher, batch_gt_instances, batch_img_metas,
                    batch_gt_instances_ignore)

        # the student use the label assignment from the teacher to learn
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, label_assignment_results,
                                     batch_data_samples)
        return losses
