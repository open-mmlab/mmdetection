# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.models.task_modules import SamplingResult
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList, OptConfigType, reduce_mean
from .fcn_mask_head import FCNMaskHead


@MODELS.register_module()
class DynamicMaskHead(FCNMaskHead):
    r"""Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_convs (int): Number of convolution layer.
            Defaults to 4.
        roi_feat_size (int): The output size of RoI extractor,
            Defaults to 14.
        in_channels (int): Input feature channels.
            Defaults to 256.
        conv_kernel_size (int): Kernel size of convolution layers.
            Defaults to 3.
        conv_out_channels (int): Output channels of convolution layers.
            Defaults to 256.
        num_classes (int): Number of classes.
            Defaults to 80
        class_agnostic (int): Whether generate class agnostic prediction.
            Defaults to False.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        upsample_cfg (:obj:`ConfigDict` or dict): The config for
            upsample layer.
        conv_cfg (:obj:`ConfigDict` or dict, optional): The convolution
            layer config.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The norm layer config.
        dynamic_conv_cfg (:obj:`ConfigDict` or dict): The dynamic convolution
            layer config.
        loss_mask (:obj:`ConfigDict` or dict): The config for mask loss.
    """

    def __init__(self,
                 num_convs: int = 4,
                 roi_feat_size: int = 14,
                 in_channels: int = 256,
                 conv_kernel_size: int = 3,
                 conv_out_channels: int = 256,
                 num_classes: int = 80,
                 class_agnostic: bool = False,
                 upsample_cfg: ConfigType = dict(
                     type='deconv', scale_factor=2),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 dynamic_conv_cfg: ConfigType = dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=14,
                     with_proj=False,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_mask: ConfigType = dict(
                     type='DiceLoss', loss_weight=8.0),
                 **kwargs) -> None:
        super().__init__(
            num_convs=num_convs,
            roi_feat_size=roi_feat_size,
            in_channels=in_channels,
            conv_kernel_size=conv_kernel_size,
            conv_out_channels=conv_out_channels,
            num_classes=num_classes,
            class_agnostic=class_agnostic,
            upsample_cfg=upsample_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            loss_mask=loss_mask,
            **kwargs)
        assert class_agnostic is False, \
            'DynamicMaskHead only support class_agnostic=False'
        self.fp16_enabled = False

        self.instance_interactive_conv = MODELS.build(dynamic_conv_cfg)

    def init_weights(self) -> None:
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)

    def forward(self, roi_feat: Tensor, proposal_feat: Tensor) -> Tensor:
        """Forward function of DynamicMaskHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_preds (Tensor): Predicted foreground masks with shape
            (batch_size*num_proposals, num_classes, pooling_h*2, pooling_w*2).
        """

        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)

        x = proposal_feat_iic.permute(0, 2, 1).reshape(roi_feat.size())

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_preds = self.conv_logits(x)
        return mask_preds

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (num_pos, num_classes, h, w).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
        """
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        num_pos = pos_labels.new_ones(pos_labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            loss_mask = self.loss_mask(
                mask_preds[torch.arange(num_pos).long(), pos_labels,
                           ...].sigmoid(),
                mask_targets,
                avg_factor=avg_factor)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)
