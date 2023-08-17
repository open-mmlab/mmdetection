from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, reduce_mean
# from mmpl.registry import MODELS, TASK_UTILS
from mmdet.registry import MODELS, TASK_UTILS
from mmengine.model import BaseModel
from einops import rearrange

# from mmpl.utils import ConfigType, OptConfigType
from mmdet.utils import ConfigType, OptConfigType


@MODELS.register_module()
class BinarySemanticSegHead(BaseModel):
    def __init__(
            self,
            num_classes=1,
            align_corners=False,
            loss_mask: ConfigType = dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: Optional[dict] = None):
        super(BinarySemanticSegHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.loss_mask = MODELS.build(loss_mask)
        if loss_dice is not None:
            self.loss_dice = MODELS.build(loss_dice)

    def forward(self, *args, **kwargs):
        pass
        return

    def loss(self,
             mask_preds: Tensor,
             seg_labels: Tensor,
             ):
        bs = mask_preds.size(0)

        # dice loss
        if hasattr(self, 'loss_dice'):
            loss_dice = self.loss_dice(mask_preds, seg_labels, avg_factor=bs)
        else:
            loss_dice = torch.zeros([]).to(mask_preds.device)

        # mask loss
        # FocalLoss support input of shape (n, num_class)
        h, w = mask_preds.shape[-2:]
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w, 1)
        mask_preds = mask_preds.reshape(-1, 1)
        # shape (num_total_gts, h, w) -> (num_total_gts * h * w)
        mask_targets = seg_labels.reshape(-1, 1)
        # target is (1 - mask_targets) !!!
        loss_mask = self.loss_mask(mask_preds, mask_targets, avg_factor=h * w)

        loss_dict = dict()
        loss_dict['loss_mask'] = loss_mask
        loss_dict['loss_dice'] = loss_dice
        return loss_dict

    def get_targets(
        self,
        cls_scores_list: List[Tensor],
        mask_preds_list: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        return_sampling_results: bool = False
    ) -> Tuple[List[Union[Tensor, int]]]:
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.
            return_sampling_results (bool): Whether to return the sampling
                results. Defaults to False.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - avg_factor (int): Average factor that is used to average\
                    the loss. When using sampling method, avg_factor is
                    usually the sum of positive and negative priors. When
                    using `MaskPseudoSampler`, `avg_factor` is usually equal
                    to the number of positive priors.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end.
        """
        results = multi_apply(self._get_targets_single, cls_scores_list,
                              mask_preds_list, batch_gt_instances,
                              batch_img_metas)
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])

        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])

        res = (labels_list, label_weights_list, mask_targets_list,
               mask_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list)

        return res + tuple(rest_results)

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image.
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image.
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image.
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_masks = gt_instances.masks
        gt_labels = gt_instances.labels

        target_shape = mask_pred.shape[-2:]
        if gt_masks.shape[0] > 0:
            gt_masks_downsampled = F.interpolate(
                gt_masks.unsqueeze(1).float(), target_shape,
                mode='nearest').squeeze(1).long()
        else:
            gt_masks_downsampled = gt_masks

        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        downsampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_masks_downsampled)
        # assign and sample # assign_result is the 1-based
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=downsampled_gt_instances,
            img_meta=img_meta)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        # 第0类为背景
        num_queries = pred_instances.scores.shape[0]
        labels = gt_labels.new_full((num_queries, ),
                                    0,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_queries)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

