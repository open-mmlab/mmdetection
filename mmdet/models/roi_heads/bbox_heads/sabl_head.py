import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_nms)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmcv.cnn import ConvModule


@HEADS.register_module
class SABLHead(nn.Module):

    def __init__(self,
                 cls_in_channels=256,
                 reg_in_channels=256,
                 roi_feat_size=7,
                 reg_feat_up_ratio=2,
                 reg_pre_kernel=3,
                 reg_pos_kernel=3,
                 reg_pre_num=2,
                 reg_pos_num=1,
                 num_classes=80,
                 cls_out_channels=1024,
                 reg_offset_out_channels=256,
                 reg_cls_out_channels=256,
                 cls_fcs_num=1,
                 reg_fcs_num=0,
                 reg_class_agnostic=True,
                 norm_cfg=None,
                 bbox_coder=dict(
                     type='BucketingBBoxCoder',
                     bucket_num=14,
                     scale_factor=1.7),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox_reg=dict(
                     type='SmoothL1Loss', beta=0.1, loss_weight=1.0)):
        super(SABLHead, self).__init__()
        self.cls_in_channels = cls_in_channels
        self.reg_in_channels = reg_in_channels
        self.roi_feat_size = roi_feat_size
        self.reg_feat_up_ratio = int(reg_feat_up_ratio)
        self.bucket_num = bbox_coder.bucket_num
        assert self.reg_feat_up_ratio // 2 >= 1
        self.up_reg_feat_size = roi_feat_size * self.reg_feat_up_ratio
        assert self.up_reg_feat_size == bbox_coder.bucket_num
        self.reg_pre_kernel = reg_pre_kernel
        self.reg_pos_kernel = reg_pos_kernel
        self.reg_pre_num = reg_pre_num
        self.reg_pos_num = reg_pos_num
        self.num_classes = num_classes
        self.cls_out_channels = cls_out_channels
        self.reg_offset_out_channels = reg_offset_out_channels
        self.reg_cls_out_channels = reg_cls_out_channels
        self.cls_fcs_num = cls_fcs_num
        self.reg_fcs_num = reg_fcs_num
        self.reg_class_agnostic = reg_class_agnostic
        assert self.reg_class_agnostic
        self.norm_cfg = norm_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cls = build_loss(loss_bbox_cls)
        self.loss_bbox_reg = build_loss(loss_bbox_reg)

        self.cls_fcs = self._add_fc_branch(self.cls_fcs_num,
                                           self.cls_in_channels,
                                           self.roi_feat_size,
                                           self.cls_out_channels)

        self.side_num = int(np.ceil(self.bucket_num / 2))

        if self.reg_feat_up_ratio > 1:
            self.upsample_x = nn.ConvTranspose1d(
                reg_in_channels,
                reg_in_channels,
                self.reg_feat_up_ratio,
                stride=self.reg_feat_up_ratio)
            self.upsample_y = nn.ConvTranspose1d(
                reg_in_channels,
                reg_in_channels,
                self.reg_feat_up_ratio,
                stride=self.reg_feat_up_ratio)

        self.reg_pre_convs = nn.ModuleList()
        for i in range(self.reg_pre_num):
            reg_pre_conv = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=reg_pre_kernel,
                padding=reg_pre_kernel // 2,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_pre_convs.append(reg_pre_conv)

        self.reg_pos_conv_xs = nn.ModuleList()
        for i in range(self.reg_pos_num):
            reg_pos_conv_x = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=(1, reg_pos_kernel),
                padding=(0, reg_pos_kernel // 2),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_pos_conv_xs.append(reg_pos_conv_x)
        self.reg_pos_conv_ys = nn.ModuleList()
        for i in range(self.reg_pos_num):
            reg_pos_conv_y = ConvModule(
                reg_in_channels,
                reg_in_channels,
                kernel_size=(reg_pos_kernel, 1),
                padding=(reg_pos_kernel // 2, 0),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU'))
            self.reg_pos_conv_ys.append(reg_pos_conv_y)

        self.reg_conv_att_x = nn.Conv2d(reg_in_channels, 1, 1)
        self.reg_conv_att_y = nn.Conv2d(reg_in_channels, 1, 1)

        self.fc_cls = nn.Linear(self.cls_out_channels, self.num_classes + 1)
        self.relu = nn.ReLU(inplace=True)

        self.reg_cls_fcs = self._add_fc_branch(self.reg_fcs_num,
                                               self.reg_in_channels, 1,
                                               self.reg_cls_out_channels)
        self.reg_offset_fcs = self._add_fc_branch(self.reg_fcs_num,
                                                  self.reg_in_channels, 1,
                                                  self.reg_offset_out_channels)
        self.fc_reg_cls = nn.Linear(self.reg_cls_out_channels, 1)
        self.fc_reg_offset = nn.Linear(self.reg_offset_out_channels, 1)

    def _add_fc_branch(self, num_branch_fcs, in_channels, roi_feat_size,
                       fc_out_channels):
        in_channels = in_channels * roi_feat_size * roi_feat_size
        branch_fcs = nn.ModuleList()
        for i in range(num_branch_fcs):
            fc_in_channels = (in_channels if i == 0 else fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, fc_out_channels))
        return branch_fcs

    def init_weights(self):
        for module_list in [
                self.reg_cls_fcs, self.reg_offset_fcs, self.cls_fcs
        ]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
        if self.reg_feat_up_ratio > 1:
            nn.init.kaiming_normal_(
                self.upsample_x.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.upsample_x.bias, 0)
            nn.init.kaiming_normal_(
                self.upsample_y.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.upsample_y.bias, 0)

        for i in range(self.reg_pre_num):
            self.reg_pre_convs[i].init_weights()
        for i in range(self.reg_pos_num):
            self.reg_pos_conv_xs[i].init_weights()
            self.reg_pos_conv_ys[i].init_weights()
        nn.init.normal_(self.reg_conv_att_x.weight, 0, 0.01)
        nn.init.constant_(self.reg_conv_att_x.bias, 0)
        nn.init.normal_(self.reg_conv_att_y.weight, 0, 0.01)
        nn.init.constant_(self.reg_conv_att_y.bias, 0)
        nn.init.normal_(self.fc_reg_offset.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg_offset.bias, 0)
        nn.init.normal_(self.fc_reg_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_reg_cls.bias, 0)
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls.bias, 0)

    def cls_forward(self, cls_x):
        cls_x = cls_x.view(cls_x.size(0), -1)
        for fc in self.cls_fcs:
            cls_x = self.relu(fc(cls_x))
        cls_score = self.fc_cls(cls_x)
        return cls_score

    def side_pool(self, reg_x):
        reg_fx = reg_x
        reg_fy = reg_x
        reg_fx_att = self.reg_conv_att_x(reg_fx).sigmoid()
        reg_fy_att = self.reg_conv_att_y(reg_fy).sigmoid()
        reg_fx_att = reg_fx_att / reg_fx_att.sum(dim=2).unsqueeze(2)
        reg_fy_att = reg_fy_att / reg_fy_att.sum(dim=3).unsqueeze(3)
        reg_fx = (reg_fx * reg_fx_att).sum(dim=2)
        reg_fy = (reg_fy * reg_fy_att).sum(dim=3)
        return reg_fx, reg_fy

    def reg_pool(self, reg_x):
        for reg_pre_conv in self.reg_pre_convs:
            reg_x = reg_pre_conv(reg_x)
        reg_fx, reg_fy = self.side_pool(reg_x)

        if self.reg_pos_num > 0:
            reg_fx = reg_fx.unsqueeze(2)
            reg_fy = reg_fy.unsqueeze(3)
            for i in range(self.reg_pos_num):
                reg_fx = self.reg_pos_conv_xs[i](reg_fx)
                reg_fy = self.reg_pos_conv_ys[i](reg_fy)
            reg_fx = reg_fx.squeeze(2)
            reg_fy = reg_fy.squeeze(3)
        if self.reg_feat_up_ratio > 1:
            reg_fx = self.relu(self.upsample_x(reg_fx))
            reg_fy = self.relu(self.upsample_y(reg_fy))
        reg_fx = torch.transpose(reg_fx, 1, 2)
        reg_fy = torch.transpose(reg_fy, 1, 2)
        return reg_fx.contiguous(), reg_fy.contiguous()

    def reg_pred(self, x, offfset_fcs, cls_fcs):
        x_offset = x.view(-1, self.reg_in_channels)
        x_cls = x.view(-1, self.reg_in_channels)

        for fc in offfset_fcs:
            x_offset = self.relu(fc(x_offset))
        for fc in cls_fcs:
            x_cls = self.relu(fc(x_cls))
        offset_pred = self.fc_reg_offset(x_offset)
        cls_pred = self.fc_reg_cls(x_cls)

        offset_pred = offset_pred.view(x.size(0), -1)
        cls_pred = cls_pred.view(x.size(0), -1)

        return offset_pred, cls_pred

    def side_flip(self, feat):
        l_end = int(np.ceil(self.up_reg_feat_size / 2))
        r_start = int(np.floor(self.up_reg_feat_size / 2))
        feat_fl = feat[:, :l_end]
        feat_fr = feat[:, r_start:].flip(dims=(1, ))
        feat_fl = feat_fl.contiguous()
        feat_fr = feat_fr.contiguous()
        feat = torch.cat([feat_fl, feat_fr], dim=-1)
        return feat

    def reg_forward(self, reg_x):
        outs = self.reg_pool(reg_x)
        edge_offset_preds = []
        edge_cls_preds = []
        reg_fx = outs[0]
        reg_fy = outs[1]
        offset_pred_x, cls_pred_x = self.reg_pred(reg_fx, self.reg_offset_fcs,
                                                  self.reg_cls_fcs)
        offset_pred_y, cls_pred_y = self.reg_pred(reg_fy, self.reg_offset_fcs,
                                                  self.reg_cls_fcs)
        offset_pred_x = self.side_flip(offset_pred_x)
        offset_pred_y = self.side_flip(offset_pred_y)
        cls_pred_x = self.side_flip(cls_pred_x)
        cls_pred_y = self.side_flip(cls_pred_y)
        edge_offset_preds = torch.cat([offset_pred_x, offset_pred_y], dim=-1)
        edge_cls_preds = torch.cat([cls_pred_x, cls_pred_y], dim=-1)

        return (edge_cls_preds, edge_offset_preds)

    def forward(self, x):

        bbox_pred = self.reg_forward(x)
        cls_score = self.cls_forward(x)

        return cls_score, bbox_pred

    def get_targets(self, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = self.bucket_target(pos_proposals, neg_proposals,
                                             pos_gt_bboxes, pos_gt_labels,
                                             rcnn_train_cfg)
        (labels, label_weights, bucket_cls_targets, bucket_cls_weights,
         bucket_offset_targets, bucket_offset_weights) = cls_reg_targets
        return (labels, label_weights, (bucket_cls_targets,
                                        bucket_offset_targets),
                (bucket_cls_weights, bucket_offset_weights))

    def bucket_target(self,
                      pos_proposals_list,
                      neg_proposals_list,
                      pos_gt_bboxes_list,
                      pos_gt_labels_list,
                      rcnn_train_cfg,
                      concat=True):
        (labels, label_weights, bucket_cls_targets, bucket_cls_weights,
         bucket_offset_targets, bucket_offset_weights) = multi_apply(
             self.bucket_target_single,
             pos_proposals_list,
             neg_proposals_list,
             pos_gt_bboxes_list,
             pos_gt_labels_list,
             cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bucket_cls_targets = torch.cat(bucket_cls_targets, 0)
            bucket_cls_weights = torch.cat(bucket_cls_weights, 0)
            bucket_offset_targets = torch.cat(bucket_offset_targets, 0)
            bucket_offset_weights = torch.cat(bucket_offset_weights, 0)
        return (labels, label_weights, bucket_cls_targets, bucket_cls_weights,
                bucket_offset_targets, bucket_offset_weights)

    def bucket_target_single(self, pos_proposals, neg_proposals, pos_gt_bboxes,
                             pos_gt_labels, cfg):
        num_pos = pos_proposals.size(0)
        num_neg = neg_proposals.size(0)
        num_samples = num_pos + num_neg
        labels = pos_gt_bboxes.new_full((num_samples, ),
                                        self.num_classes,
                                        dtype=torch.long)
        label_weights = pos_proposals.new_zeros(num_samples)
        bucket_cls_targets = pos_proposals.new_zeros(num_samples,
                                                     4 * self.side_num)
        bucket_cls_weights = pos_proposals.new_zeros(num_samples,
                                                     4 * self.side_num)
        bucket_offset_targets = pos_proposals.new_zeros(
            num_samples, 4 * self.side_num)
        bucket_offset_weights = pos_proposals.new_zeros(
            num_samples, 4 * self.side_num)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            label_weights[:num_pos] = 1.0
            (pos_bucket_offset_targets, pos_bucket_offset_weights,
             pos_bucket_cls_targets,
             pos_bucket_cls_weights) = self.bbox_coder.encode(
                 pos_proposals, pos_gt_bboxes)
            bucket_cls_targets[:num_pos, :] = pos_bucket_cls_targets
            bucket_cls_weights[:num_pos, :] = pos_bucket_cls_weights
            bucket_offset_targets[:num_pos, :] = pos_bucket_offset_targets
            bucket_offset_weights[:num_pos, :] = pos_bucket_offset_weights
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return (labels, label_weights, bucket_cls_targets, bucket_cls_weights,
                bucket_offset_targets, bucket_offset_weights)

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bucket_cls_preds, bucket_offset_preds = bbox_pred
            bucket_cls_targets, bucket_offset_targets = bbox_targets
            bucket_cls_weights, bucket_offset_weights = bbox_weights
            # edge cls
            bucket_cls_preds = bucket_cls_preds.view(-1, self.side_num)
            bucket_cls_targets = bucket_cls_targets.view(-1, self.side_num)
            bucket_cls_weights = bucket_cls_weights.view(-1, self.side_num)
            losses['loss_bbox_cls'] = self.loss_bbox_cls(
                bucket_cls_preds,
                bucket_cls_targets,
                bucket_cls_weights,
                avg_factor=bucket_cls_targets.size(0),
                reduction_override=reduction_override)

            losses['loss_bbox_reg'] = self.loss_bbox_reg(
                bucket_offset_preds,
                bucket_offset_targets,
                bucket_offset_weights,
                avg_factor=bucket_offset_targets.size(0),
                reduction_override=reduction_override)

        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes, confids = self.bbox_coder.decode(rois[:, 1:], bbox_pred,
                                                     img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            confids = None
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=confids)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            edge_cls_preds, edge_offset_preds = bbox_preds
            edge_cls_preds_ = edge_cls_preds[inds]
            edge_offset_preds_ = edge_offset_preds[inds]
            bbox_pred_ = [edge_cls_preds_, edge_offset_preds_]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if rois.size(1) == 4:
            new_rois, _ = self.bbox_coder.decode(rois, bbox_pred,
                                                 img_meta['img_shape'])
        else:
            bboxes, _ = self.bbox_coder.decode(rois[:, 1:], bbox_pred,
                                               img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
