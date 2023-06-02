# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, bbox_project
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from ..utils.misc import unpack_gt_instances
from .semi_base import SemiBaseDetector


@MODELS.register_module()
class SoftTeacher(SemiBaseDetector):
    r"""Implementation of `End-to-End Semi-Supervised Object Detection
    with Soft Teacher <https://arxiv.org/abs/2106.09018>`_

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

        x = self.student.extract_feat(batch_inputs)

        losses = {}
        rpn_losses, rpn_results_list = self.rpn_loss_by_pseudo_instances(
            x, batch_data_samples)
        losses.update(**rpn_losses)
        losses.update(**self.rcnn_cls_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples, batch_info))
        losses.update(**self.rcnn_reg_loss_by_pseudo_instances(
            x, rpn_results_list, batch_data_samples))
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        return rename_loss_dict('unsup_',
                                reweight_loss_dict(losses, unsup_weight))

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        x = self.teacher.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.teacher.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        batch_data_samples = filter_gt_instances(
            batch_data_samples,
            score_thr=self.semi_train_cfg.pseudo_label_initial_score_thr)

        reg_uncs_list = self.compute_uncertainty_with_aug(
            x, batch_data_samples)

        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)

        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': []
        }
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)
        return batch_data_samples, batch_info

    def rpn_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                     batch_data_samples: SampleList) -> dict:
        """Calculate rpn loss from a batch of inputs and pseudo data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
        Returns:
            dict: A dictionary of rpn loss components
        """

        rpn_data_samples = copy.deepcopy(batch_data_samples)
        rpn_data_samples = filter_gt_instances(
            rpn_data_samples, score_thr=self.semi_train_cfg.rpn_pseudo_thr)
        proposal_cfg = self.student.train_cfg.get('rpn_proposal',
                                                  self.student.test_cfg.rpn)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.student.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        for key in rpn_losses.keys():
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        return rpn_losses, rpn_results_list

    def rcnn_cls_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                          unsup_rpn_results_list: InstanceList,
                                          batch_data_samples: SampleList,
                                          batch_info: dict) -> dict:
        """Calculate classification loss from a batch of inputs and pseudo data
        samples.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            unsup_rpn_results_list (list[:obj:`InstanceData`]):
                List of region proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn
                classification loss components
        """
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        cls_data_samples = copy.deepcopy(batch_data_samples)
        cls_data_samples = filter_gt_instances(
            cls_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)

        outputs = unpack_gt_instances(cls_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(cls_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')
            assign_result = self.student.roi_head.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        selected_bboxes = [res.priors for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(x, rois)
        # cls_reg_targets is a tuple of labels, label_weights,
        # and bbox_targets, bbox_weights
        cls_reg_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, self.student.train_cfg.rcnn)

        selected_results_list = []
        for bboxes, data_samples, teacher_matrix, teacher_img_shape in zip(
                selected_bboxes, batch_data_samples,
                batch_info['homography_matrix'], batch_info['img_shape']):
            student_matrix = torch.tensor(
                data_samples.homography_matrix, device=teacher_matrix.device)
            homography_matrix = teacher_matrix @ student_matrix.inverse()
            projected_bboxes = bbox_project(bboxes, homography_matrix,
                                            teacher_img_shape)
            selected_results_list.append(InstanceData(bboxes=projected_bboxes))

        with torch.no_grad():
            results_list = self.teacher.roi_head.predict_bbox(
                batch_info['feat'],
                batch_info['metainfo'],
                selected_results_list,
                rcnn_test_cfg=None,
                rescale=False)
            bg_score = torch.cat(
                [results.scores[:, -1] for results in results_list])
            # cls_reg_targets[0] is labels
            neg_inds = cls_reg_targets[
                0] == self.student.roi_head.bbox_head.num_classes
            # cls_reg_targets[1] is label_weights
            cls_reg_targets[1][neg_inds] = bg_score[neg_inds].detach()

        losses = self.student.roi_head.bbox_head.loss(
            bbox_results['cls_score'], bbox_results['bbox_pred'], rois,
            *cls_reg_targets)
        # cls_reg_targets[1] is label_weights
        losses['loss_cls'] = losses['loss_cls'] * len(
            cls_reg_targets[1]) / max(sum(cls_reg_targets[1]), 1.0)
        return losses

    def rcnn_reg_loss_by_pseudo_instances(
            self, x: Tuple[Tensor], unsup_rpn_results_list: InstanceList,
            batch_data_samples: SampleList) -> dict:
        """Calculate rcnn regression loss from a batch of inputs and pseudo
        data samples.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            unsup_rpn_results_list (list[:obj:`InstanceData`]):
                List of region proposals.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.

        Returns:
            dict[str, Tensor]: A dictionary of rcnn
                regression loss components
        """
        rpn_results_list = copy.deepcopy(unsup_rpn_results_list)
        reg_data_samples = copy.deepcopy(batch_data_samples)
        for data_samples in reg_data_samples:
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.reg_uncs <
                    self.semi_train_cfg.reg_pseudo_thr]
        roi_losses = self.student.roi_head.loss(x, rpn_results_list,
                                                reg_data_samples)
        return {'loss_bbox': roi_losses['loss_bbox']}

    def compute_uncertainty_with_aug(
            self, x: Tuple[Tensor],
            batch_data_samples: SampleList) -> List[Tensor]:
        """Compute uncertainty with augmented bboxes.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.

        Returns:
            list[Tensor]: A list of uncertainty for pseudo bboxes.
        """
        auged_results_list = self.aug_box(batch_data_samples,
                                          self.semi_train_cfg.jitter_times,
                                          self.semi_train_cfg.jitter_scale)
        # flatten
        auged_results_list = [
            InstanceData(bboxes=auged.reshape(-1, auged.shape[-1]))
            for auged in auged_results_list
        ]

        self.teacher.roi_head.test_cfg = None
        results_list = self.teacher.roi_head.predict(
            x, auged_results_list, batch_data_samples, rescale=False)
        self.teacher.roi_head.test_cfg = self.teacher.test_cfg.rcnn

        reg_channel = max(
            [results.bboxes.shape[-1] for results in results_list]) // 4
        bboxes = [
            results.bboxes.reshape(self.semi_train_cfg.jitter_times, -1,
                                   results.bboxes.shape[-1])
            if results.bboxes.numel() > 0 else results.bboxes.new_zeros(
                self.semi_train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for results in results_list
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel,
                             4)[torch.arange(bbox.shape[0]), label]
                for bbox, label in zip(bboxes, labels)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel,
                            4)[torch.arange(unc.shape[0]), label]
                for unc, label in zip(box_unc, labels)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0)
                     for bbox in bboxes]
        box_unc = [
            torch.mean(
                unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4), dim=-1)
            if wh.numel() > 0 else unc for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(batch_data_samples, times, frac):
        """Augment bboxes with jitter."""

        def _aug_single(box):
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2,
                                                          2).reshape(-1, 4))
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device) *
                aug_scale[None, ...])
            new_box = box.clone()[None, ...].expand(times, box.shape[0],
                                                    -1) + offset
            return new_box

        return [
            _aug_single(data_samples.gt_instances.bboxes)
            for data_samples in batch_data_samples
        ]
