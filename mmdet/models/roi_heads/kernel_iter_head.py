# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_attention)
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import build_assigner, build_sampler, multi_apply
from mmdet.core.bbox.samplers import MaskPseudoSampler
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_head, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads import BaseRoIHead


@HEADS.register_module()
class KernelUpdateHead(BaseModule):

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 conv_kernel_size=3,
                 feat_transform_cfg=None,
                 hard_mask_thr=0.5,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 mask_upsample_stride=1,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 kernel_updator_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_rank=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0)):
        super(KernelUpdateHead, self).__init__()
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout
        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]

        self.kernel_update_conv = build_attention(kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
        else:
            self.feat_transform = None

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))

        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

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
             x (Tensor): feature of imgs, shape (N,C,H,W).
             proposal_feat (Tensor): Kernel weight of conv, \
                 with shape (N,num_proposal+num_stuff,C,K,K)
             mask_preds (Tensor): predict mask, \
                 with shape (N,num_proposal+num_stuff,H,W)
        Return: a tuple containing the following targets.
             -cls_score (Tensor): predict classes, \
                 with shape (N,num_proposal+num_stuff,C)
             -new_mask_preds (Tensor): new predict mask, \
                 with shape (N,num_proposal+num_stuff,H,W)\
             -obj_feat (Tensor): updated proposal_feat, \
                 with shape (N,num_proposal+num_stuff,C,K,K)
        """
        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()
        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)
        # obj_feat in shape [N, num_proposal+stuff, C, K, K]
        # -> [N,  num_proposal+stuff, C, K*K]
        # -> [N,  num_proposal+stuff, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)
        obj_feat = self.kernel_update_conv(
            x_feat, proposal_feat)  # (bs*(num_proposal+stuff)),K*K,c)
        # [N, num_proposal+stuff, K*K, C] -> [N, num_proposal+stuff, K*K*C]
        # -> [num_proposal+stuff, N, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(
            self.attention(obj_feat))  # (num_proposal+num_stuff,N,K*K*C)
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(
            1, 0, 2)  # (bs,num_proposal+num_stuff,c*k*k) k is always 1
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)
        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))
        cls_feat = obj_feat.sum(-2)  # (N,num_proposal+num_stuff,1,c)
        mask_feat = obj_feat
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        cls_score = self.fc_cls(cls_feat).view(
            N, num_proposals, -1)  # (bs,um_proposal+num_stuff,num_classes)
        # [N, um_proposal+num_stuff, K*K, C]
        # -> [N, um_proposal+num_stuff, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        # new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        return cls_score, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
            N, num_proposals, self.in_channels, self.conv_kernel_size,
            self.conv_kernel_size)

    @force_fp32(apply_to=('cls_score', 'mask_pred'))
    def loss(self,
             object_feats,
             cls_score,
             mask_pred,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos).clamp_(min=1.0)

        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        assert mask_pred.shape[0] == cls_score.shape[0]
        assert mask_pred.shape[1] == cls_score.shape[1]

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score.view(num_preds, -1),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(
                    cls_score.view(num_preds, -1)[pos_inds], labels[pos_inds])
        if mask_pred is not None:
            bool_pos_inds = pos_inds.type(torch.bool)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            H, W = mask_pred.shape[-2:]
            if pos_inds.any():
                pos_mask_pred = mask_pred.reshape(num_preds, H,
                                                  W)[bool_pos_inds]
                pos_mask_targets = mask_targets[bool_pos_inds]
                losses['loss_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
                losses['loss_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

                if self.loss_rank is not None:
                    batch_size = mask_pred.size(0)
                    rank_target = mask_targets.new_full((batch_size, H, W),
                                                        self.ignore_label,
                                                        dtype=torch.long)
                    rank_inds = pos_inds.view(batch_size,
                                              -1).nonzero(as_tuple=False)
                    batch_mask_targets = mask_targets.view(
                        batch_size, -1, H, W).bool()
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j
                    losses['loss_rank'] = self.loss_rank(
                        mask_pred, rank_target, ignore_index=self.ignore_label)
            else:
                losses['loss_mask'] = mask_pred.sum() * 0
                losses['loss_dice'] = mask_pred.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_pred.sum() * 0

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
        label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples, H, W)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            pos_mask_targets = pos_gt_mask
            mask_targets[pos_inds, ...] = pos_mask_targets
            mask_weights[pos_inds, ...] = 1

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_mask.new_full((self.num_stuff_classes, ),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_weights = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_mask.device)
            sem_thing_weights = pos_mask.new_zeros(
                (self.num_stuff_classes, self.num_thing_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1

            label_weights[:, self.num_thing_classes:] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

        return labels, label_weights, mask_targets, mask_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    gt_labels,
                    rcnn_train_cfg,
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

        labels, label_weights, mask_targets, mask_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
        return labels, label_weights, mask_targets, mask_weights

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      test_cfg, img_meta):
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg.mask_thr
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores):
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bbox_result, segm_result


@HEADS.register_module()
class KernelIterHead(BaseRoIHead):

    def __init__(self,
                 num_stages=6,
                 assign_stages=5,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 merge_cls_scores=False,
                 do_panoptic=False,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=0,
                 mask_head=dict(
                     type='KernelUpdateHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_out_stride=4,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.merge_cls_scores = merge_cls_scores
        self.mask_out_stride = mask_out_stride
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_thing_classes + num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.num_proposals = num_proposals
        super(KernelIterHead, self).__init__(
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(
                    self.mask_sampler[stage], MaskPseudoSampler), \
                    'Sparse Mask only support `MaskPseudoSampler`'

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))

    def _mask_forward(self, stage, x, object_feats, mask_preds):
        """
        Args:
            stage (int): indice use which mask head.\
            x (Tensor): feature map, shape(N,C,H,W).\
            object_feats (Tensor): Kernel weight, \
                with shape (N,num_proposal,C,K,K).\
            mask_preds (Tensor): predict mask, shape (N,num_proposal,H,W).\
        Return:
            tuple: a tuple containing the following targets.
                - cls_score (Tensor): predict classes,\
                    with shape (B,num_proposal_num_stuff,classes).\
                - mask_preds (Tensor): predict mask, \
                    with shape (B,num_proposal_num_stuff,H,W).\
                - scaled_mask_preds (Tensor): predict scaled mask,\
                    with shape (B,num_proposal_num_stuff,2*H,2*W).\
                - object_feats (Tensor): Kernel weight , \
                    with shape (B,num_proposal_num_stuff,C,K,K)
        """
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats = mask_head(
            x, object_feats, mask_preds)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds
        mask_results = dict(
            cls_score=cls_score,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)

        return mask_results

    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_sem_seg=None,
                      gt_sem_cls=None):
        """
        Args:
            x (Tensor): Feature map, shape (N,C,H/8,W/8)
            proposal_feats (Tensor): Kernel weight, \
                with shape(N,num_proposal+stuff,C,K,K)
            mask_preds (Tensor): predict mask,\
                with shape (N,num_proposal+stuff,C,H/8,W/8)
            cls_score (NoneType)
            img_metas (List[Dict]): information of imgs
            gt_masks List[Tensor]: gt_mask, each with shape (num_gt,H,W)
            gt_labels:  List[Tensor]: gt_label, each with shape (num_gt)
            gt_sem_seg List[Tensor]: gt stuff segmentation mask,\
                each with each with shape (num_stuff,h,w)
            gt_sem_cls List[Tensor]: gt_stuff label,\
                each with shape (num_stuff,)
        Return:
            all_stage_loss (Dict): loss of each stage loss

        """
        num_imgs = len(img_metas)
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()

        if cls_score is not None:
            prev_cls_score = cls_score.detach()
        else:
            prev_cls_score = [None] * num_imgs

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']
            sampling_results = []
            assign_results = []
            for i in range(num_imgs):
                if stage < self.assign_stages:
                    mask_for_assign = prev_mask_preds[i][:self.num_proposals]
                    if prev_cls_score[i] is not None:
                        cls_for_assign = prev_cls_score[
                            i][:self.num_proposals, :self.num_thing_classes]
                    else:
                        cls_for_assign = None
                    assign_result = self.mask_assigner[stage].assign(
                        cls_for_assign, mask_for_assign, gt_labels[i],
                        gt_masks[i], img_metas[i])
                    assign_results.append(assign_result)
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], scaled_mask_preds[i], gt_masks[i])
                sampling_results.append(sampling_result)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                gt_masks,
                gt_labels,
                self.train_cfg[stage],
                True,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls)

            single_stage_loss = self.mask_head[stage].loss(
                object_feats,
                cls_score,
                scaled_mask_preds,
                *mask_targets,
            )
            for key, value in single_stage_loss.items():
                all_stage_loss[f's{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]

            prev_mask_preds = scaled_mask_preds.detach()
            prev_cls_score = cls_score.detach()

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    rescale=False):

        # Decode initial proposals
        num_imgs = len(img_metas)
        # num_proposals = proposal_feats.size(1)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id])
                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=True)
                mask_indices = topk_indices // num_classes
                labels_per_img = topk_indices % num_classes
                masks_per_img = scaled_mask_preds[img_id][mask_indices]
                single_result = self.mask_head[-1].get_seg_masks(
                    masks_per_img, labels_per_img, scores_per_img,
                    self.test_cfg, img_metas[img_id])
                results.append(single_result)
        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('SparseMask does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        C, H, W = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, C, -1)).view(
            num_imgs, num_proposals, H, W)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

    def get_panoptic(self, cls_scores, mask_preds, test_cfg, img_meta):
        # resize mask predictions back
        scores = cls_scores[:self.num_proposals][:, :self.num_thing_classes]
        thing_scores, thing_labels = scores.max(dim=1)
        stuff_scores = cls_scores[
            self.num_proposals:][:, self.num_thing_classes:].diag()
        stuff_labels = torch.arange(
            0, self.num_stuff_classes) + self.num_thing_classes
        stuff_labels = stuff_labels.to(thing_labels.device)

        total_masks = self.mask_head[-1].rescale_masks(mask_preds, img_meta)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        panoptic_result = self.merge_stuff_thing(total_masks, total_labels,
                                                 total_scores,
                                                 test_cfg.merge_stuff_thing)
        return dict(pan_results=panoptic_result)

    def merge_stuff_thing(self,
                          total_masks,
                          total_labels,
                          total_scores,
                          merge_cfg=None):

        H, W = total_masks.shape[-2:]
        panoptic_seg = total_masks.new_full((H, W),
                                            self.num_classes,
                                            dtype=torch.long)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg.overlap_thr:
                    continue

                panoptic_seg[mask] = total_labels[k] \
                    + current_segment_id * INSTANCE_OFFSET
                current_segment_id += 1

        return panoptic_seg.cpu().numpy()
