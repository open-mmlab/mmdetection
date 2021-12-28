import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import mask_target
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.utils import build_transformer
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
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
        upsample_cfg (dict): The config for upsample layer.
        conv_cfg (dict): The convolution layer config.
        norm_cfg (dict): The norm layer config.
        dynamic_conv_cfg (dict): The dynamic convolution layer config.
        loss_mask (dict): The config for mask loss.
    """

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 upsample_cfg=dict(type='deconv', scale_factor=2),
                 conv_cfg=None,
                 norm_cfg=None,
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=14,
                     with_proj=False,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_mask=dict(type='DiceLoss', loss_weight=8.0),
                 **kwargs):
        super(DynamicMaskHead, self).__init__(
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

        self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat):
        """Forward function of DynamicMaskHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_pred (Tensor): Predicted foreground masks with shape
                (batch_size*num_proposals, num_classes,
                                        pooling_h*2, pooling_w*2).
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
        mask_pred = self.conv_logits(x)
        return mask_pred

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        num_pos = labels.new_ones(labels.size()).float().sum()
        avg_factor = torch.clamp(reduce_mean(num_pos), min=1.).item()
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            loss_mask = self.loss_mask(
                mask_pred[torch.arange(num_pos).long(), labels, ...].sigmoid(),
                mask_targets,
                avg_factor=avg_factor)
        loss['loss_mask'] = loss_mask
        return loss

    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):

        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets
