# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

import torch
from mmengine.structures import PixelData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class TwoStagePanopticSegmentor(TwoStageDetector):
    """Base class of Two-stage Panoptic Segmentor.

    As well as the components in TwoStageDetector, Panoptic Segmentor has extra
    semantic_head and panoptic_fusion_head.
    """

    def __init__(
            self,
            backbone: ConfigType,
            neck: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            # for panoptic segmentation
            semantic_head: OptConfigType = None,
            panoptic_fusion_head: OptConfigType = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if semantic_head is not None:
            self.semantic_head = MODELS.build(semantic_head)

        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

            self.num_things_classes = self.panoptic_fusion_head.\
                num_things_classes
            self.num_stuff_classes = self.panoptic_fusion_head.\
                num_stuff_classes
            self.num_classes = self.panoptic_fusion_head.num_classes

    @property
    def with_semantic_head(self) -> bool:
        """bool: whether the detector has semantic head"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    @property
    def with_panoptic_fusion_head(self) -> bool:
        """bool: whether the detector has panoptic fusion head"""
        return hasattr(self, 'panoptic_fusion_head') and \
            self.panoptic_fusion_head is not None

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
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        semantic_loss = self.semantic_head.loss(x, batch_data_samples)
        losses.update(semantic_loss)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            List[:obj:`DetDataSample`]: Return the packed panoptic segmentation
                results of input images. Each DetDataSample usually contains
                'pred_panoptic_seg'. And the 'pred_panoptic_seg' has a key
                ``sem_seg``, which is a tensor of shape (1, h, w).
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        seg_preds = self.semantic_head.predict(x, batch_img_metas, rescale)

        results_list = self.panoptic_fusion_head.predict(
            results_list, seg_preds)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    # TODO the code has not been verified and needs to be refactored later.
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head``, ``roi_head`` and
                ``semantic_head`` forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)
        rpn_outs = self.rpn_head.forward(x)
        results = results + (rpn_outs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            rpn_results_list = self.rpn_head.predict_by_feat(
                *rpn_outs, batch_img_metas=batch_img_metas, rescale=False)
        else:
            # TODO: Not checked currently.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # roi_head
        roi_outs = self.roi_head(x, rpn_results_list)
        results = results + (roi_outs)

        # semantic_head
        sem_outs = self.semantic_head.forward(x)
        results = results + (sem_outs['seg_preds'], )

        return results

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[PixelData]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`]): The
                annotation data of every samples.
            results_list (List[PixelData]): Panoptic segmentation results of
                each image.

        Returns:
            List[:obj:`DetDataSample`]: Return the packed panoptic segmentation
                results of input images. Each DetDataSample usually contains
                'pred_panoptic_seg'. And the 'pred_panoptic_seg' has a key
                ``sem_seg``, which is a tensor of shape (1, h, w).
        """

        for data_sample, pred_panoptic_seg in zip(data_samples, results_list):
            data_sample.pred_panoptic_seg = pred_panoptic_seg
        return data_samples
