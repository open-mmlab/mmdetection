# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import BaseDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) /
                     (mask_pred_.sum([1, 2]) + 1e-6))


@MODELS.register_module()
class SparseInst(BaseDetector):
    """Implementation of `SparseInst <https://arxiv.org/abs/1912.02424>`_

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        encoder (:obj:`ConfigDict` or dict): The encoder module.
        decoder (:obj:`ConfigDict` or dict): The decoder module.
        criterion (:obj:`ConfigDict` or dict, optional): The training matcher
            and losses. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of SparseInst. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: ConfigType,
                 backbone: ConfigType,
                 encoder: ConfigType,
                 decoder: ConfigType,
                 criterion: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # backbone
        self.backbone = MODELS.build(backbone)
        # encoder & decoder
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        # matcher & loss (matcher is built in loss)
        self.criterion = MODELS.build(criterion)

        # inference
        self.cls_threshold = test_cfg.score_thr
        self.mask_threshold = test_cfg.mask_thr_binary

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        results = self.decoder(x)
        return results

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
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        max_shape = batch_inputs.shape[-2:]
        output = self._forward(batch_inputs)

        pred_scores = output['pred_logits'].sigmoid()
        pred_masks = output['pred_masks'].sigmoid()
        pred_objectness = output['pred_scores'].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        results_list = []
        for batch_idx, (scores_per_image, mask_pred_per_image,
                        datasample) in enumerate(
                            zip(pred_scores, pred_masks, batch_data_samples)):
            result = InstanceData()
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.labels = labels
                results_list.append(result)
                continue

            img_meta = datasample.metainfo
            # rescoring mask using maskness
            scores = rescoring_mask(scores,
                                    mask_pred_per_image > self.mask_threshold,
                                    mask_pred_per_image)
            h, w = img_meta['img_shape'][:2]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1),
                size=max_shape,
                mode='bilinear',
                align_corners=False)[:, :, :h, :w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_pred_per_image = F.interpolate(
                    mask_pred_per_image,
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            result.masks = mask_pred
            result.scores = scores
            result.labels = labels
            # create an empty bbox in InstanceData to avoid bugs when
            # calculating metrics.
            result.bboxes = result.scores.new_zeros(len(scores), 4)
            results_list.append(result)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self._forward(batch_inputs)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)

        losses = self.criterion(outs, batch_gt_instances, batch_img_metas,
                                batch_gt_instances_ignore)
        return losses

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        return x
