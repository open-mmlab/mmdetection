# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..losses import GIoULoss, QualityFocalLoss
from .semi_base import SemiBaseDetector


@MODELS.register_module()
class DenseTeacher(SemiBaseDetector):
    r"""Implementation of `Dense Teacher: Dense Pseudo-Labels for
    Semi-supervised Object Detection <https://arxiv.org/abs/2207.02541v2>`_

    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.quality_focal_loss = QualityFocalLoss(reduction='sum')
        self.iou_loss = GIoULoss()

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        self.teacher.eval()
        dense_predicts = self.teacher(batch_inputs)
        batch_info = {}
        batch_info['dense_predicts'] = dense_predicts
        return batch_data_samples, batch_info

    @staticmethod
    def permute_to_N_HWA_K(tensor: Tensor, K: int) -> Tensor:
        """Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA),
        K)"""

        assert tensor.dim() == 4, tensor.shape
        N, _, H, W = tensor.shape
        tensor = tensor.view(N, -1, K, H, W)
        tensor = tensor.permute(0, 3, 4, 1, 2)
        tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
        return tensor

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        burn_in_steps = self.semi_train_cfg.get('burn_in_steps', 5000)
        if self.iter <= burn_in_steps:
            return {}
        else:
            teacher_logits, teacher_deltas, teacher_quality = batch_info[
                'dense_predicts']
            student_logits, student_deltas, student_quality = self.student(
                batch_inputs)

            num_classes = self.student.bbox_head.cls_out_channels

            student_logits = torch.cat([
                self.permute_to_N_HWA_K(x, num_classes) for x in student_logits
            ],
                                       dim=1).view(-1, num_classes)
            teacher_logits = torch.cat([
                self.permute_to_N_HWA_K(x, num_classes) for x in teacher_logits
            ],
                                       dim=1).view(-1, num_classes)

            student_deltas = torch.cat(
                [self.permute_to_N_HWA_K(x, 4) for x in student_deltas],
                dim=1).view(-1, 4)
            teacher_deltas = torch.cat(
                [self.permute_to_N_HWA_K(x, 4) for x in teacher_deltas],
                dim=1).view(-1, 4)

            student_quality = torch.cat(
                [self.permute_to_N_HWA_K(x, 1) for x in student_quality],
                dim=1).view(-1, 1)
            teacher_quality = torch.cat(
                [self.permute_to_N_HWA_K(x, 1) for x in teacher_quality],
                dim=1).view(-1, 1)

            with torch.no_grad():
                # Region Selection
                ratio = self.semi_train_cfg.get('k_ratio', 0.01)
                count_num = int(teacher_logits.size(0) * ratio)
                teacher_probs = teacher_logits.sigmoid()
                max_vals = torch.max(teacher_probs, 1)[0]
                sorted_vals, sorted_inds = torch.topk(max_vals,
                                                      teacher_logits.size(0))
                mask = torch.zeros_like(max_vals)
                mask[sorted_inds[:count_num]] = 1.
                fg_num = sorted_vals[:count_num].sum()
                b_mask = mask > 0.

            loss_logits = self.quality_focal_loss(
                student_logits, teacher_logits, weight=mask) / fg_num

            loss_deltas = self.iou_loss(student_deltas[b_mask],
                                        teacher_deltas[b_mask])

            loss_quality = F.binary_cross_entropy(
                student_quality[b_mask].sigmoid(),
                teacher_quality[b_mask].sigmoid(),
                reduction='mean')

            losses = {
                'distill_loss_logits':
                self.semi_train_cfg.get('logits_weight', 1.) * loss_logits,
                'distill_loss_quality':
                self.semi_train_cfg.get('quality_weight', 1.) * loss_quality,
                'distill_loss_deltas':
                self.semi_train_cfg.get('deltas_weight', 1.) * loss_deltas,
                'fore_ground_sum':
                fg_num,
            }

            unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
            target = burn_in_steps * 2
            if self.iter <= target:
                unsup_weight *= (self.iter -
                                 burn_in_steps) / self.burn_in_steps

            return rename_loss_dict('unsup_',
                                    reweight_loss_dict(losses, unsup_weight))
