# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_project
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBaseDetector

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None


@MODELS.register_module()
class ConsistentTeacher(SemiBaseDetector):
    r"""Implementation of `Consistent-Teacher: Towards Reducing Inconsistent
    Pseudo-targets in Semi-supervised Object Detection
    <https://arxiv.org/abs/2209.01589>`_

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

        if self.semi_train_cfg is not None:
            self.freeze(self.teacher)
            self.unsup_weight = self.semi_train_cfg.unsup_weight

        num_classes = self.teacher.bbox_head.num_classes
        num_scores = self.semi_train_cfg.num_scores

        self.register_buffer('scores', torch.zeros((num_classes, num_scores)))
        self.iter = 0

    def set_iter(self, step):
        self.iter = step

    # def loss(self, multi_batch_inputs: Dict[str, Tensor],
    #          multi_batch_data_samples: Dict[str, SampleList]) -> dict:
    #     """Calculate losses from multi-branch inputs and data samples.

    #     Args:
    #         multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
    #             input images, each value with shape (N, C, H, W).
    #             Each value should usually be mean centered and std scaled.
    #         multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
    #             The dict of multi-branch data samples.

    #     Returns:
    #         dict: A dictionary of loss components
    #     """
    #     losses = dict()
    #     losses.update(**self.loss_by_gt_instances(
    #         multi_batch_inputs['sup'], multi_batch_data_samples['sup']))

    #     origin_pesudo_data_samples, batch_info = self.get_pseudo_instances(
    #         multi_batch_inputs['unsup_teacher'],
    #         multi_batch_data_samples['unsup_teacher'])
    #     multi_batch_data_samples[
    #         'unsup_student'] = self.project_pseudo_instances(
    #             origin_pesudo_data_samples,
    #             multi_batch_data_samples['unsup_student'])
    #     losses.update(**self.loss_by_pseudo_instances(
    #         multi_batch_inputs['unsup_student'],
    #         multi_batch_data_samples['unsup_student'], batch_info))

    #     return losses

    def loss_by_gt_instances(self, batch_inputs: Tensor,
                             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """

        losses = self.student.loss(batch_inputs, batch_data_samples)
        losses['num_gts'] = torch.tensor(
            sum([
                len(data_samples.gt_instances)
                for data_samples in batch_data_samples
            ]) / len(batch_data_samples))
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        return rename_loss_dict('sup_', reweight_loss_dict(losses, sup_weight))

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

        unsup_weight = self.unsup_weight
        if self.iter < self.semi_train_cfg.get('warmup_step', -1):
            unsup_weight = 0

        losses = {}
        bbox_losses, bbox_results_list = self.bbox_loss_by_pseudo_instances(
            x, batch_data_samples)
        losses.update(**bbox_losses)
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

        bbox_results_list = [
            data_sample.proposals for data_sample in batch_data_samples
        ]
        results_list = self.teacher.bbox_head.predict(
            x, bbox_results_list, rescale=False)
        results_label_list = [r[1] for r in results_list]

        # dynamic thresholds
        thrs = []
        for _, data_samples in enumerate(batch_data_samples):
            dynamic_ratio = self.semi_train_cfg.dynamic_ratio
            scores = data_samples[:, 4].clone()
            scores = scores.sort(descending=True)[0]
            if len(scores) == 0:
                thrs.append(1)  # no kept pseudo boxes
            else:
                # num_gt = int(scores.sum() + 0.5)
                num_gt = int(scores.sum() * dynamic_ratio + 0.5)
                num_gt = min(num_gt, len(scores) - 1)
                thrs.append(scores[num_gt] - 1e-5)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=(thr for thr in thrs))

        scores = torch.cat([
            data_samples.gt_instances[:, 4]
            for data_samples in batch_data_samples
        ])
        labels = torch.cat(results_label_list)
        thrs = torch.zeros_like(scores)
        for label in torch.unique(labels):
            label = int(label)
            scores_add = (scores[labels == label])
            num_buffers = len(self.scores[label])
            scores_new = torch.cat(
                [scores_add.float(), self.scores[label].float()])[:num_buffers]
            self.scores[label] = scores_new
            thr = self.gmm_policy(
                scores_new[scores_new > 0],
                given_gt_thr=self.semi_train_cfg.get('given_gt_thr', 0),
                policy=self.semi_train_cfg.get('policy', 'high'))
            thrs[labels == label] = thr
        mean_thr = thrs.mean()
        if len(thrs) == 0:
            mean_thr.fill_(0)

        thrs = torch.split(
            thrs, [len(data_samples) for data_samples in batch_data_samples])

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results
            data_samples.gt_instances.bboxes = bbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(
                    self.data_preprocessor.device), data_samples.ori_shape)
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=(thr_tmp for thr_tmp in thrs))

        batch_info = {
            'feat': x,
            'img_shape': [],
            'homography_matrix': [],
            'metainfo': [],
        }
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(
                torch.from_numpy(data_samples.homography_matrix).to(
                    self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)

        return batch_data_samples, batch_info

    def bbox_loss_by_pseudo_instances(self, x: Tuple[Tensor],
                                      batch_data_samples: SampleList) -> dict:
        """Calculate bbox loss from a batch of inputs and pseudo data samples.

        Args:
            x (tuple[Tensor]): Features from bbox_head.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
        Returns:
            dict: A dictionary of rpn loss components
        """
        bbox_data_samples = copy.deepcopy(batch_data_samples)
        num_bbox_data = [len(bbox) for bbox in bbox_data_samples]
        bbox_losses, bbox_results_list = self.student.bbox_head.loss_and_predict(
            x, bbox_data_samples)
        if len([n for n in num_bbox_data if n > 0]) < len(num_bbox_data) / 2:
            bbox_losses = reweight_loss_dict(
                bbox_losses,
                weight=self.semi_train_cfg.get('background_weight', 1e-2))
        bbox_losses['num_bbox_data'] = torch.tensor(sum([]))
        return bbox_losses, bbox_results_list

    def gmm_policy(self,
                   scores,
                   given_gt_thr: float = 0.5,
                   policy: str = 'high'):
        """The policy of choosing pseudo label.

        GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the found bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.
        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]

        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment
                            == 1) & (scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr
