# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import (ATTENTION, FEEDFORWARD_NETWORK,
                                         TRANSFORMER_LAYER,
                                         build_transformer_layer)
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy


@TRANSFORMER_LAYER.register_module()
class KernelUpdator(BaseModule):
    """Dynamic Kernel Updator in Kernel Update Head.

    Args:
        in_channels (int): The number of channels of input feature map.
            Default: 256.
        feat_channels (int): The number of middle-stage channels in
            the kernel updator. Default: 64.
        out_channels (int): The number of output channels.
        gate_sigmoid (bool): Whether use sigmoid function in gate
            mechanism. Default: True.
        gate_norm_act (bool): Whether add normalization and activation
            layer in gate mechanism. Default: False.
        activate_out (bool): Whether add activation after gate mechanism.
            Default: False.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU').
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='LN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.dynamic_layer = nn.Linear(self.in_channels,
                                       self.feat_channels * 2)
        self.input_layer = nn.Linear(self.in_channels, self.feat_channels * 2)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels)

        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(self.norm_cfg)[1]

        self.norm_in = build_norm_layer(self.norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(self.norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(self.norm_cfg,
                                              self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(self.norm_cfg,
                                               self.feat_channels)[1]

        self.activation = build_activation_layer(self.act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = build_norm_layer(self.norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature):
        """Forward function of KernelUpdator.

        Args:
            update_feature (torch.Tensor): Feature map assembled from
                each group. It would be reshaped with last dimension
                shape: `self.in_channels`.
            input_feature (torch.Tensor): Intermediate feature
                with shape: (N, num_classes, conv_kernel_size**2, channels).
        Returns:
            Tensor: The output tensor of shape (N*C1/C2, K*K, C2), where N is
            the number of classes, C1 and C2 are the feature map channels of
            KernelUpdateHead and KernelUpdator, respectively.
        """
        update_feature = update_feature.reshape((-1, self.in_channels))
        num_proposals = update_feature.shape[0]
        # dynamic_layer works for
        # phi_1 and psi_3 in Eq.(4) and (5) of K-Net paper
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.feat_channels].view(
            (-1, self.feat_channels))
        param_out = parameters[:, self.feat_channels:].view(
            (-1, self.feat_channels))

        # input_layer works for
        # phi_2 and psi_4 in Eq.(4) and (5) of K-Net paper
        input_feats = self.input_layer(
            input_feature.reshape((num_proposals, -1, self.feat_channels)))
        input_in = input_feats[..., :self.feat_channels]
        input_out = input_feats[..., self.feat_channels:]

        # `gate_feats` is F^G in K-Net paper
        gate_feats = input_in * param_in.unsqueeze(-2)
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()

        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)
        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # Gate mechanism. Eq.(5) in original paper.
        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out
        features = self.activation(self.fc_norm(self.fc_layer(features)))

        return features


@HEADS.register_module()
class KernelUpdateHead(BaseModule):
    """Kernel update head.

    Args:
        in_channels (int): Number of channels in the input feature
            map. Defaults to 256.
        out_channels (int): Number of channels in the output feature
            map. Defaults to 256.
        num_things_classes (int): Number of things classes. Defaults to 80.
        num_stuff_classes (int): Number of stuff classes. Defaults to 53.
        ignore_label (int): Class indice to be ignored. Defaults to 255.
        num_cls_fcs (int): Number of fully connected layers
            in the classification branch. Defaults to 1.
        num_mask_fcs (int): Number of convolution layers in the
            mask branch. Defaults to 1.
        act_cfg (:obj:`mmcv.Config`): Config of activation layer.
            Defaults to dict(type='ReLU', inplace=True).
        conv_kernel_size (int): Kernel size. Defaults to 3.
        feat_transform_cfg (:obj:`mmcv.Config`): Config of feature transform.
            Defaults to None.
        mask_upsample_stride (int): Upsample stride of mask. Defaults to 2.
        kernel_updator_cfg (:obj:`mmcv.Config`): Config of kernel updator.
            Defaults to dict( type='DynamicConv', ...).
        attn_cfg (:obj:`mmcv.Config`): Config of attention. Defaults to
            dict( type='MultiheadAttention', ...).
        ffn_cfg (:obj:`mmcv.Config`): Config of FFN.
            Defaults to dict( type='FFN', ...).
        attn_ffn_norm_cfg (:obj:`mmcv.Config`): Config of normalization layer
            for attention and ffn layer. Defaults to dict(type='LN').
        loss_rank (:obj:`mmcv.Config`): Config of rank loss. Defaults to None.
        loss_mask (:obj:`mmcv.Config`): Config of mask loss.
            Defaults to dict( type='CrossEntropyLoss', ...).
        loss_dice (:obj:`mmcv.Config`): Config of dice loss. Defaults to
            dict(type='DiceLoss', ...).
        loss_cls (:obj:`mmcv.Config`): Config of classification loss. Defaults
            to dict(type='FocalLoss', ...).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            in_channels=256,
            out_channels=256,
            num_things_classes=80,
            num_stuff_classes=53,
            ignore_label=255,
            num_cls_fcs=1,
            num_mask_fcs=1,
            act_cfg=dict(type='ReLU', inplace=True),
            conv_kernel_size=1,
            feat_transform_cfg=None,
            mask_upsample_stride=2,
            kernel_updator_cfg=dict(
                type='DynamicConv',
                in_channels=256,
                feat_channels=64,
                out_channels=256,
                input_feat_shape=1,
                act_cfg=dict(type='ReLU', inplace=True),
                norm_cfg=dict(type='LN')),
            # attention + ffn + norm
            attn_cfg=dict(
                type='MultiheadAttention',
                embed_dims=256,
                num_heads=8,
                attn_drop=0.0),
            ffn_cfg=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                dropout=0.0),
            attn_ffn_norm_cfg=dict(type='LN'),
            loss_rank=None,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            loss_dice=dict(type='DiceLoss', loss_weight=3.0),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.ignore_label = ignore_label

        self.num_cls_fcs = num_cls_fcs
        self.num_mask_fcs = num_mask_fcs
        self.act_cfg = act_cfg
        self.conv_kernel_size = conv_kernel_size
        self.feat_transform_cfg = feat_transform_cfg
        self.mask_upsample_stride = mask_upsample_stride

        self.kernel_updator_cfg = kernel_updator_cfg
        self.attn_cfg = attn_cfg
        self.ffn_cfg = ffn_cfg
        self.attn_ffn_norm_cfg = attn_ffn_norm_cfg

        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.loss_cls = build_loss(loss_cls)

        self._init_layers()

    def _init_layers(self):
        self.attention = ATTENTION.build(self.attn_cfg)
        self.attention_norm = build_norm_layer(
            self.attn_ffn_norm_cfg,
            self.in_channels * self.conv_kernel_size**2)[1]

        self.kernel_update_conv = build_transformer_layer(
            self.kernel_updator_cfg)

        if self.feat_transform_cfg is not None:
            kernel_size = self.feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                self.in_channels,
                self.in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                **self.feat_transform_cfg)
        else:
            self.feat_transform = None

        self.ffn = FEEDFORWARD_NETWORK.build(self.ffn_cfg)
        self.ffn_norm = build_norm_layer(self.attn_ffn_norm_cfg,
                                         self.in_channels)[1]

        self.cls_fcs = nn.Sequential()
        for _ in range(self.num_cls_fcs):
            self.cls_fcs.add_module(
                str(len(self.cls_fcs)),
                nn.Linear(self.in_channels, self.in_channels, bias=False))
            self.cls_fcs.add_module(
                str(len(self.cls_fcs)),
                build_norm_layer(dict(type='LN'), self.in_channels)[1])
            self.cls_fcs.add_module(
                str(len(self.cls_fcs)), build_activation_layer(self.act_cfg))

        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(self.in_channels, self.num_classes + 1)

        self.mask_fcs = nn.Sequential()
        for _ in range(self.num_mask_fcs):
            self.mask_fcs.add_module(
                str(len(self.mask_fcs)),
                nn.Linear(self.in_channels, self.in_channels, bias=False))
            self.mask_fcs.add_module(
                str(len(self.mask_fcs)),
                build_norm_layer(dict(type='LN'), self.in_channels)[1])
            self.mask_fcs.add_module(
                str(len(self.mask_fcs)), build_activation_layer(self.act_cfg))

        self.fc_mask = nn.Linear(self.in_channels, self.out_channels)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass

        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

    def forward(self, x, proposal_feat, mask_preds):
        """
        Args:
            x (Tensor): Feature maps, shape (batch_size, c, h, w).
            proposal_feats (Tensor): Proposal feats, shape (batch_size,
                n, c, kernel_size, kernel_size). n is (num_proposals +
                num_stuff_classes) for panoptic segmentation, num_proposals for
                instance segmentation.
            mask_preds (Tensor): Mask logits, shape (batch_size, n, h, w).

        Returns:
            tuple[Tensor]:

            - cls_scores (Tensor): Classification logits, shape (batch_size,
              n, cls_channels). n is (num_proposals + num_stuff) for panoptic
              segmentation, num_proposals for instance segmentation.
            - new_mask_preds (Tensor): Mask logits, shape (batch_size,
              n, h, w).
            - obj_feat (Tensor): Proposal features, shape (batch_size,
              n, c, k, k).
        """
        # Group Feature Assembling

        c, h, w = x.shape[-3:]
        batch_size, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != h or mask_w != w:
            gather_mask = F.interpolate(
                mask_preds, (h, w), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > 0.5
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # Adaptive Feature Update

        # shape (batch_size, num_proposals, c, k, k) ->
        # (batch_size, num_proposals, c, k * k) ->
        # (batch_size, num_proposals, k * k, c)
        proposal_feat = proposal_feat.reshape(
            (batch_size, num_proposals, self.in_channels,
             -1)).permute(0, 1, 3, 2)
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # Kernel Interaction

        # shape (batch_size, num_proposals, k * k, c) ->
        # (batch_size, num_proposals, k * k * c) ->
        # (num_proposals, batch_size, k * k *c)
        obj_feat = obj_feat.reshape(
            (batch_size, num_proposals, -1)).permute(1, 0, 2)

        obj_feat = self.attention_norm(self.attention(obj_feat))

        # shape (num_proposals, batch_size, k * k * c) ->
        # (batch_size, num_proposals, k * k * c) ->
        # (batch_size, num_proposals, k * k, c)
        obj_feat = obj_feat.permute(1, 0, 2).reshape(
            (batch_size, num_proposals, -1, self.in_channels))
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        # shape (batch_size, num_proposals, c)
        cls_feat = obj_feat.sum(-2)
        cls_feat = self.cls_fcs(cls_feat)

        # shape (batch_size, num_proposals, k * k, c)
        mask_feat = obj_feat
        mask_feat = self.mask_fcs(mask_feat)

        # shape (batch_size, num_proposals, cls_channels)
        cls_scores = self.fc_cls(cls_feat)

        # shape (batch_size, num_proposals, k * k, c) ->
        # (batch_size, num_proposals, c, k * k)
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        # shape (batch_size, num_proposals, c, k * k) ->
        # (batch_size, num_proposals, c, k, k)
        mask_feat = mask_feat.reshape(
            (batch_size, num_proposals, c, self.conv_kernel_size,
             self.conv_kernel_size))

        new_mask_preds = []
        for i in range(batch_size):
            new_mask_preds.append(
                # shape (1, num_proposals, h, w)
                F.conv2d(
                    # shape (1, c, h, w)
                    x[i:i + 1],
                    # shape (num_proposals, c, k, k)
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))
        # shape (batch_size, num_proposals, h, w)
        new_mask_preds = torch.cat(new_mask_preds, dim=0)

        # shape (num_proposals, batch_size, k * k *c) ->
        # (batch_size, num_proposals, c, k, k)
        obj_feat = obj_feat.permute(0, 1, 3, 2).reshape(
            (batch_size, num_proposals, self.in_channels,
             self.conv_kernel_size, self.conv_kernel_size))

        return cls_scores, new_mask_preds, obj_feat

    def _get_targets_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                            pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                            cfg):
        """Compute classification and segmentation targets for single image.

        Args:
            pos_inds (Tensor): Indice of positive samples.
            neg_inds (Tensor): Indice of negative samples.
            pos_mask (Tensor): Positive masks.
            neg_mask (Tensor): Negative masks.
            pos_gt_mask (Tensor): Positive ground truth instance masks.
            pos_gt_labels (Tensor): Positive groud truth instance class
                indices.
            gt_sem_seg (Tensor): Ground truth stuff masks.
            gt_sem_cls (Tensor): Ground truth stuff class indices.
            cfg (:obj:`mmcv.Config`): Training config for RCNN.

        Returns:
            tuple:

            - labels (Tensor): shape (n,). n is (num_proposals +
              num_stuff_classes) for panoptic segmentation,
              num_proposals for instance segmentation.
            - label_weights (Tensor): shape (n,).
            - mask_targets (Tensor): shape (n, h, w).
        """
        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg
        h, w = pos_mask.shape[-2:]

        labels = pos_mask.new_full((num_samples, ),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        mask_targets = pos_mask.new_zeros((num_samples, h, w))

        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            mask_targets[pos_inds, ...] = pos_gt_mask

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_mask.new_full((self.num_stuff_classes, ),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_mask.new_zeros((self.num_stuff_classes, h, w))
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_mask.device)
            sem_thing_weights = pos_mask.new_zeros(
                (self.num_stuff_classes, self.num_things_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)

            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_things_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg

            label_weights[:, self.num_things_classes:] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])

        return labels, label_weights, mask_targets

    def get_targets(self, sampling_results, rcnn_train_cfg, gt_sem_seg,
                    gt_sem_cls):
        """Compute classification and segmentation targets for all images.

        Args:
            sampling_results (list[:obj:`MaskSamplingResult`]): Mask sampling
                results.
            rcnn_train_cfg (:obj:`mmcv.Config`): Training config for RCNN.
            gt_sem_seg (list[Tensor]): Ground truth stuff class indices
                for all images. Each with shape (n, ), n is the number
                of stuff class in a image.
            gt_sem_cls (list[Tensor]): Ground truth mask of stuff
                for all images. Each with shape (n, h, w).

        Returns:
            tuple:

            - labels (Tensor): shape (batch_size * n,). n is
              (num_proposals + num_stuff_classes) for panoptic segmentation,
              num_proposals for instance segmentation.
            - label_weights (Tensor): shape (batch_size * n,).
            - mask_targets (Tensor): shape (batch_size * n, h, w).
        """
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]

        labels, label_weights, mask_targets = multi_apply(
            self._get_targets_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rcnn_train_cfg)

        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        mask_targets = torch.cat(mask_targets, 0)

        return labels, label_weights, mask_targets

    @force_fp32(apply_to=('cls_scores', 'mask_preds'))
    def loss(self, cls_scores, mask_preds, labels, label_weights,
             mask_targets):
        """Loss function.

        Args:
            cls_scores (Tensor): Classification logits, shape (batch_size,
                n, cls_channels). n is (num_proposals + num_stuff_classes)
                for panoptic segmentation, num_proposals for instance
                segmentation.
            mask_preds (Tensor): Mask logits, shape (batch_size, n, h, w).
            labels (Tensor): Label targets, shape (batch_size * n).
            label_weights (Tensor): Label weights, shape (batch_size * n).
            mask_targets (Tensor): Mask targets, shape (batch_size * n, h, w).

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
        batch_size, num_proposals, h, w = mask_preds.shape
        num_preds = batch_size * num_proposals

        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos).clamp_(min=1.0)

        if cls_scores is not None:
            if cls_scores.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_scores.view(num_preds, -1),
                    labels,
                    label_weights,
                    avg_factor=avg_factor)
                losses['pos_acc'] = accuracy(
                    cls_scores.view(num_preds, -1)[pos_inds], labels[pos_inds])

        if mask_preds is not None:
            bool_pos_inds = pos_inds.type(torch.bool)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform mask loss for BG anymore.
            if pos_inds.any():
                pos_mask_pred = mask_preds.reshape(
                    (num_preds, h, w))[bool_pos_inds]
                pos_mask_targets = mask_targets[bool_pos_inds]
                losses['loss_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
                losses['loss_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

                if self.loss_rank is not None:
                    rank_target = mask_targets.new_full((batch_size, h, w),
                                                        self.ignore_label,
                                                        dtype=torch.long)
                    rank_inds = pos_inds.view(
                        (batch_size, -1)).nonzero(as_tuple=False)
                    batch_mask_targets = mask_targets.view(
                        (batch_size, -1, h, w)).bool()
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j
                    losses['loss_rank'] = self.loss_rank(
                        mask_preds,
                        rank_target,
                        ignore_index=self.ignore_label)
            else:
                losses['loss_mask'] = mask_preds.sum() * 0
                losses['loss_dice'] = mask_preds.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_preds.sum() * 0

        return losses
