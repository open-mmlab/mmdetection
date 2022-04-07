# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import BaseModule

from mmdet.core import build_assigner, build_sampler, multi_apply
from ..builder import HEADS, build_loss, build_neck


@HEADS.register_module()
class ConvKernelHead(BaseModule):

    def __init__(self,
                 num_proposals=100,
                 in_channels=256,
                 out_channels=256,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_seg_convs=1,
                 num_loc_convs=1,
                 att_dropout=False,
                 localization_fpn=None,
                 conv_kernel_size=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 semantic_fpn=True,
                 train_cfg=None,
                 xavier_init_kernel=False,
                 kernel_init_std=0.01,
                 proposal_feats_with_obj=False,
                 loss_mask=None,
                 loss_seg=None,
                 loss_cls=None,
                 loss_dice=None,
                 loss_rank=None,
                 feat_downsample_stride=1,
                 feat_refine=True,
                 conv_normal_init=False,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cat_stuff_mask=False,
                 **kwargs):
        super(ConvKernelHead, self).__init__()
        self.num_proposals = num_proposals
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.in_channels = in_channels
        self.proposal_feats_with_obj = proposal_feats_with_obj
        self.sampling = False
        self.localization_fpn = build_neck(localization_fpn)
        self.semantic_fpn = semantic_fpn
        self.norm_cfg = norm_cfg
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.conv_kernel_size = conv_kernel_size
        self.xavier_init_kernel = xavier_init_kernel
        self.kernel_init_std = kernel_init_std
        self.feat_downsample_stride = feat_downsample_stride
        self.conv_normal_init = conv_normal_init
        self.feat_refine = feat_refine
        self.num_loc_convs = num_loc_convs
        self.num_seg_convs = num_seg_convs
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.out_channels = out_channels
        if cat_stuff_mask:
            self.num_classes = num_thing_classes + num_stuff_classes
        else:
            self.num_classes = num_thing_classes
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg
        self.cat_stuff_mask = cat_stuff_mask

        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)
        else:
            self.loss_mask = loss_mask

        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = loss_dice

        if loss_seg is not None:
            self.loss_seg = build_loss(loss_seg)
        else:
            self.loss_seg = loss_seg
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = loss_cls

        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='MaskPseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_kernels = nn.Conv2d(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias=False)

        if self.semantic_fpn:
            if self.loss_seg.use_sigmoid:
                self.conv_seg = nn.Conv2d(self.out_channels, self.num_classes,
                                          1)
            else:
                self.conv_seg = nn.Conv2d(self.out_channels,
                                          self.num_classes + 1, 1)

        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg)
            self.seg_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg)

        self.loc_convs = nn.ModuleList()
        for i in range(self.num_loc_convs):
            self.loc_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.seg_convs = nn.ModuleList()
        for i in range(self.num_seg_convs):
            self.seg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        self.localization_fpn.init_weights()

        if self.feat_downsample_stride > 1 and self.conv_normal_init:
            for conv in [self.loc_convs, self.seg_convs]:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.01)

        if self.semantic_fpn:
            bias_seg = bias_init_with_prob(0.01)
            if self.loss_seg.use_sigmoid:
                normal_init(self.conv_seg, std=0.01, bias=bias_seg)
            else:
                normal_init(self.conv_seg, mean=0, std=0.01)
        if self.xavier_init_kernel:
            nn.init.xavier_uniform_(self.init_kernels.weight)
        else:
            normal_init(self.init_kernels, mean=0, std=self.kernel_init_std)

    def _decode_init_proposals(self, img, img_metas):
        """
        Args:
            img (List[Tensor]):feature map of imgs, each is \
                a 4D-tensor with shape (num_imgs,C,H_i,W_i)
            img_metas (Dict): List of image information.
        Returns: tuple: a tuple contains four elements.
                - proposal_feats (Tensor): kernel for each\
                    proposal . Each is a 4D-tensor with shape\
                    (N, num_proposal, C, 1,1).\
                    Note `cls_out_channels` should includes background.
                - x_feats (Tensor): feature for each img\
                    layer. Each with shape (N, C,H, W).
                - mask_preds (Tensor): Predict mask. Each with \
                    shape (N, num_proposal,H, W)
                - cls_scores (NoneType):
                - seg_preds (Tensor): Predict segmentation map. Each with \
                    shape (N,classes,H,W)
        """
        num_imgs = len(img_metas)

        localization_feats = self.localization_fpn(img)
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats)
        if self.feat_downsample_stride > 1 and self.feat_refine:
            loc_feats = self.ins_downsample(loc_feats)
        mask_preds = self.init_kernels(loc_feats)  # (N,num_proposal,H,W)

        if self.semantic_fpn:  # use for panoptic predict
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats)
            if self.feat_downsample_stride > 1 and self.feat_refine:
                semantic_feats = self.seg_downsample(
                    semantic_feats)  # (N, C, H/8, W/8)
        else:
            semantic_feats = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(
                semantic_feats)  # (N, thing+stuff, H, W). predict for seg
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone(
        )  # (num_proposal,c,1,1)
        proposal_feats = proposal_feats[None].expand(
            num_imgs, *proposal_feats.size()
        )  # (num_proposal,c,1,1) ->(num_imgs,num_proposal,c,1,1)

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats
        else:
            x_feats = loc_feats

        if self.proposal_feats_with_obj:  # True for panoptic
            sigmoid_masks = mask_preds.sigmoid()  # (N, num_proposal, H, W)
            nonzero_inds = sigmoid_masks > 0.5
            sigmoid_masks = nonzero_inds.float()
            obj_feats = torch.einsum(
                'bnhw,bchw->bnc', sigmoid_masks,
                x_feats)  # (N, num_proposal, C) Group Feature Assembling

        cls_scores = None

        if self.proposal_feats_with_obj:
            proposal_feats = proposal_feats + obj_feats.view(
                num_imgs, self.num_proposals, self.out_channels, 1,
                1)  # (, num_proposal, C, 1, 1)

        if self.cat_stuff_mask and not self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds

    def forward_train(self,
                      feats,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_sem_seg=None,
                      gt_sem_cls=None):
        """
        Args:
            x (List[Tensor]): each of shape (N, C, H_i, W_i) 
                encoding input images. Typically these should be 
                mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_masks (list[BitmapMasks]): true segmentation masks for each mask
                used if the architecture supports a segmentation task.
            gt_labels (list[Tensor]): classes indices 
                corresponding to each mask.
            gt_semantic_seg (list[tensor]): semantic segmentation mask 
                of stuff for images.
            gt_sem_cls (list[Tensor]): classes of each semantic_seg mask

        Returns:
            Tuple: a tuple containing the following targets.
             - losses (Dict): loss of this head
             - proposal_feats (Tensor): Kernel weight of predict , 
                shape (N,num_proposal+num_stuff,C,K,K).
             - x_feats (Tensor): Feature map, shape (N,C,H/8,W/8).
             - mask_preds (Tensor): predict mask, 
                shape (N,num_proposal+num_stuff,H/8,W/8)
             - cls_scores (None)
        """
        num_imgs = len(img_metas)
        results = self._decode_init_proposals(feats, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = results
        # proposal_feats:(N, C, H, W),x_feats:(N, C, H/8, W/8),
        # mask_preds(N, num_proposal, H/8, W/8),
        # cls_scores:None
        # seg_preds:(N,classes,H/8,W/8)
        if self.feat_downsample_stride > 1:
            # (N,num_proposal,H/8.W/8) -> (N,num_proposal,H/4.W/4)
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride,
                mode='bilinear',
                align_corners=False)
            if seg_preds is not None:
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)

        else:
            scaled_mask_preds = mask_preds
            scaled_seg_preds = seg_preds

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds,
                           gt_masks, gt_labels, gt_sem_seg, gt_sem_cls,
                           img_metas)

        if self.cat_stuff_mask and self.training:
            mask_preds = torch.cat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], dim=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = torch.cat([proposal_feats, stuff_kernels], dim=1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores

    def loss(self,
             mask_preds,
             cls_scores,
             seg_preds,
             gt_masks,
             gt_labels,
             gt_sem_seg,
             gt_sem_cls,
             img_metas,
             reduction_override=None,
             **kwargs):
        """Loss function.

        Args:
            mask_preds (Tensor): mask scores for proposal \
                with shape (batch_size, num_queries,h,w).
            cls_scores (Tensor): None
            seg_preds (Tensor): Predict class indices for each image \
                with shape (batch_size,num_classes,h,w ).
                numclasses is the sum of number of thing type
            gt_masks (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w). n is the sum of number of stuff type
                and number of instance in a image.
            gt_labels: (list[Tensor]): each with shape (n,)
            gt_sem_seg:(List[Tensor]):
            gt_sem_cls:(List[Tensor])
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        if cls_scores is None:
            detached_cls_scores = [None] * num_imgs
        else:
            detached_cls_scores = cls_scores.detach()
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.assigner.assign(detached_cls_scores[i],
                                                 mask_preds[i].detach(),
                                                 gt_labels[i], gt_masks[i],
                                                 img_metas[i])
            sampling_result = self.sampler.sample(assign_result, mask_preds[i],
                                                  gt_masks[i])
            sampling_results.append(sampling_result)

        labels, label_weights, mask_targets, mask_weights,\
            seg_targets = self.get_targets(
             sampling_results,
             gt_masks,
             self.train_cfg,
             concat=True,
             gt_sem_seg=gt_sem_seg,
             gt_sem_cls=gt_sem_cls)
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_preds = mask_preds.shape[0] * mask_preds.shape[1]

        bool_pos_inds = pos_inds.type(torch.bool)
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        # do not perform bounding box regression for BG anymore.
        H, W = mask_preds.shape[-2:]
        if pos_inds.any():
            pos_mask_pred = mask_preds.reshape(num_preds, H, W)[bool_pos_inds]
            pos_mask_targets = mask_targets[bool_pos_inds]
            losses['loss_rpn_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
            losses['loss_rpn_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

            if self.loss_rank is not None:
                batch_size = mask_preds.size(0)
                rank_target = mask_targets.new_full((batch_size, H, W),
                                                    self.ignore_label,
                                                    dtype=torch.long)
                rank_inds = pos_inds.view(batch_size,
                                          -1).nonzero(as_tuple=False)
                batch_mask_targets = mask_targets.view(batch_size, -1, H,
                                                       W).bool()
                for i in range(batch_size):
                    curr_inds = (rank_inds[:, 0] == i)
                    curr_rank = rank_inds[:, 1][curr_inds]
                    for j in curr_rank:
                        rank_target[i][batch_mask_targets[i][j]] = j
                losses['loss_rpn_rank'] = self.loss_rank(
                    mask_preds, rank_target, ignore_index=self.ignore_label)

        else:
            losses['loss_rpn_mask'] = mask_preds.sum() * 0
            losses['loss_rpn_dice'] = mask_preds.sum() * 0
            if self.loss_rank is not None:
                losses['loss_rank'] = mask_preds.sum() * 0

        if seg_preds is not None:  # (bs,classes,h,w)
            if self.loss_seg.use_sigmoid:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.view(-1, cls_channel, H * W).permute(
                    0, 2, 1).reshape(-1, cls_channel)  # (bs*h*w,classes)
                flatten_seg_target = seg_targets.view(-1)
                num_dense_pos = (flatten_seg_target >= 0) & (
                    flatten_seg_target < bg_class_ind)
                num_dense_pos = num_dense_pos.sum().float().clamp(min=1.0)
                losses['loss_rpn_seg'] = self.loss_seg(
                    flatten_seg, flatten_seg_target, avg_factor=num_dense_pos)
            else:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.view(-1, cls_channel, H * W).permute(
                    0, 2, 1).reshape(-1, cls_channel)
                flatten_seg_target = seg_targets.view(-1)
                losses['loss_rpn_seg'] = self.loss_seg(flatten_seg,
                                                       flatten_seg_target)

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                           cfg):
        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_mask.new_full((num_samples, ),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_mask.new_zeros(num_samples)
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples, H, W)
        seg_targets = pos_mask.new_full((H, W),
                                        self.num_classes,
                                        dtype=torch.long)

        if gt_sem_cls is not None and gt_sem_seg is not None:
            gt_sem_seg = gt_sem_seg.bool()
            for sem_mask, sem_cls in zip(gt_sem_seg, gt_sem_cls):
                seg_targets[sem_mask] = sem_cls.long()

        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            mask_targets[pos_inds, ...] = pos_gt_mask
            mask_weights[pos_inds, ...] = 1
            for i in range(num_pos):
                seg_targets[pos_gt_mask[i].bool()] = pos_gt_labels[i]

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, mask_targets, mask_weights, seg_targets

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    rpn_train_cfg,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_sem_seg is None:
            gt_sem_seg = [None] * 2
            gt_sem_cls = [None] * 2
        results = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rpn_train_cfg)
        (labels, label_weights, mask_targets, mask_weights,
         seg_targets) = results
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
            seg_targets = torch.stack(seg_targets, 0)
        return labels, label_weights, mask_targets, mask_weights, seg_targets

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)
