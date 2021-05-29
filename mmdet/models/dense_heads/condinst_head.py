import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, kaiming_init
from mmcv.runner import force_fp32
from mmcv.ops.nms import batched_nms

from mmdet.core import (distance2bbox, multi_apply, bbox_overlaps,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from ..losses import cross_entropy, accuracy
import pycocotools.mask as mask_util

INF = 1e8
EPS = 1e-12

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   multi_kernels,
                   multi_points,
                   multi_strides,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    num_classes = multi_scores.size(1) - 1
    bboxes = multi_bboxes[:, None].expand(
        multi_scores.size(0), num_classes, 4)
    kernels = multi_kernels[:, None].expand(
        multi_scores.size(0), num_classes, 169)
    bboxes = multi_bboxes[:, None].expand(
        multi_scores.size(0), num_classes, 4)
    points = multi_points[:, None].expand(
        multi_scores.size(0), num_classes, 2)
    strides = multi_strides[:, None].expand(
        multi_scores.size(0), num_classes)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    kernels = kernels.reshape(-1, 169)
    points = points.reshape(-1, 2)
    strides = strides.reshape(-1, 1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels, kernels, points, strides = \
        bboxes[inds], scores[inds], labels[inds], kernels[inds], points[inds], strides[inds]
    if inds.numel() == 0:
       return bboxes, labels, kernels, points, strides

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    return dets, labels[keep], kernels[keep], points[keep], strides[keep]

def dice_coefficient(x, target):
    eps = 1e-5
    n_instance = x.size(0)
    x = x.reshape(n_instance, -1)
    target = target.reshape(n_instance, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0,
        w * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shifts_y = torch.arange(0,
        h * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]

@HEADS.register_module()
class CondInstHead(AnchorFreeHead):
    """Conditional Convolutions for Instance Segmentation
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 **kwargs):
        super(CondInstHead, self).__init__(num_classes, in_channels, **kwargs)
        self.strides = strides
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_on_bbox = norm_on_bbox

        # fcos
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.fcos_reg = nn.Conv2d(
            self.feat_channels,
            4,
            3,
            padding=1)
        self.fcos_centerness = nn.Conv2d(
            self.feat_channels,
            1,
            3,
            padding=1)
        self.controller = nn.Conv2d(
            self.feat_channels,
            169,
            3,
            padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        # mask branch
        self.mask_refine = nn.ModuleList()
        in_features = ['p3', 'p4', 'p5']
        for in_feature in in_features:
            conv_block = []
            conv_block.append(
                nn.Conv2d(self.feat_channels,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())
            conv_block = nn.Sequential(*conv_block)
            self.mask_refine.append(conv_block)
        # mask head
        tower = []
        for i in range(self.stacked_convs):
            conv_block = []
            conv_block.append(
                nn.Conv2d(128,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())

            conv_block = nn.Sequential(*conv_block)
            tower.append(conv_block)

        tower.append(
            nn.Conv2d(128,
                      8,
                      kernel_size=1,
                      stride=1))
        self.mask_head = nn.Sequential(*tower)

        # conditional convs
        self.weight_nums = [80, 64, 8]
        self.bias_nums = [8, 8, 1]
        self.mask_out_stride = 4

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        kaiming_init(self.mask_refine)
        kaiming_init(self.mask_head)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    4.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        kernel_preds = []
        for i, (x,  scale) in enumerate(zip(feats, self.scales)):
            cls_feat = x
            reg_feat = x

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.fcos_cls(cls_feat)
            bbox_pred = scale(self.fcos_reg(reg_feat)).float()
            if self.norm_on_bbox:
                bbox_pred = F.relu(bbox_pred) * self.strides[i]
            else:
                bbox_pred = bbox_pred.exp()
            centerness = self.fcos_centerness(reg_feat)
            kernel_pred = self.controller(reg_feat)

            # mask feat
            if i == 0:
                mask_feat = self.mask_refine[i](x)
            elif i <= 2:
                x_p = self.mask_refine[i](x)
                target_h, target_w = mask_feat.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                mask_feat = mask_feat + x_p

            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)
            centernesses.append(centerness)
            kernel_preds.append(kernel_pred)

        mask_feat = self.mask_head(mask_feat)

        return cls_scores, bbox_preds, centernesses, kernel_preds, mask_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             kernel_preds,
             mask_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_masks=None,):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        device = cls_scores[0].device

        points_list, strides_list = self.get_points(
            featmap_sizes, bbox_preds[0].dtype, device=device)

        cls_reg_targets = self.get_targets(
            points_list,
            gt_bboxes,
            gt_labels)

        (labels_list, bbox_targets_list, gt_inds_list) = cls_reg_targets
        # gt mask
        gt_masks_list = []
        for i in range(len(gt_labels)):
            gt_label = gt_labels[i]
            gt_masks_list.append(
                torch.from_numpy(
                    np.array(gt_masks[i], dtype=np.float32)).to(gt_label.device))

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in points_list])
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(5):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list]))

        flatten_labels = torch.cat(concat_lvl_labels)
        flatten_bbox_targets = torch.cat(concat_lvl_bbox_targets)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero(as_tuple=False).squeeze(1)

        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        # classification loss
        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels,
            avg_factor=num_pos)
        if len(pos_inds) > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_bbox_pred = flatten_bbox_preds[pos_inds]
            pos_points = flatten_points[pos_inds]
            pos_centerness = flatten_centerness[pos_inds]

            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_decode_bbox_pred = distance2bbox(
                pos_points, pos_bbox_pred)
            pos_decode_bbox_targets = distance2bbox(
                pos_points, pos_bbox_targets)

            # centerness weighted iou loss
            centerness_denorm = max(
                reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                pos_centerness_targets,
                avg_factor=num_pos)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = torch.tensor(0).cuda()
        # loss mask
        loss_mask = 0
        num_mask = 0
        flatten_kernel_preds = [
            kernel.permute(0, 2, 3, 1).reshape(num_imgs, -1, 169) for kernel in kernel_preds
        ]
        flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)
        for i in range(num_imgs):
            flatten_labels_i = torch.cat(labels_list[i])
            pos_inds = ((flatten_labels_i >= 0)
                    & (flatten_labels_i < self.num_classes)).nonzero(as_tuple=False).squeeze(1)
            # mask feat
            mask_feat = mask_feats[i]
            bbox_pred_list = [
                bbox_preds[level][i].permute(1, 2, 0).reshape(-1, 4).detach()
                for level in range(5)
            ]
            bbox_pred = torch.cat(bbox_pred_list)[pos_inds]
            points = torch.cat(points_list)[pos_inds]
            pos_det_bboxes = distance2bbox(points, bbox_pred)
            idx_gt = gt_inds_list[i]
            mask_head_params = flatten_kernel_preds[i][pos_inds]
            strides = torch.cat(strides_list)[pos_inds]

            if pos_det_bboxes.shape[0] == 0 or gt_masks_list[i].shape[0] == 0:
                loss_mask += pos_det_bboxes[:, 0].sum() * 0
                continue

            # mask loss
            num_instance = len(points)
            mask_head_inputs = self.relative_coordinate_feature_generator(
                mask_feat,
                points,
                strides)
            weights, biases = parse_dynamic_params(
                mask_head_params,
                8,
                self.weight_nums,
                self.bias_nums)
            mask_logits = self.mask_heads_forward(
                mask_head_inputs,
                weights,
                biases,
                num_instance)
            mask_logits = mask_logits.reshape(-1, 1, mask_feat.size(1), mask_feat.size(2)).squeeze(1)
            # pad gt mask
            img_h, img_w = mask_feat.size(1) * 8, mask_feat.size(2) * 8
            h, w = gt_masks_list[i].size()[1:]
            gt_mask = F.pad(gt_masks_list[i], (0, img_w - w, 0, img_h - h), "constant", 0)
            start = int(8 // 2)
            gt_mask = gt_mask[:, start::8, start::8]
            gt_mask = gt_mask.gt(0.5).float()
            gt_mask = torch.index_select(gt_mask, 0, idx_gt).contiguous()
            loss_mask += dice_coefficient(mask_logits.sigmoid(), gt_mask).sum()
            num_mask += len(idx_gt)

        loss_mask = loss_mask / num_mask

        if loss_mask == 0:
            loss_mask = pos_det_bboxes[:, 0].sum() * 0

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_mask=loss_mask)

    def mask_heads_forward(self, features, weights, biases, num_instances):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                         w,
                         bias=b,
                         stride=1,
                         padding=0,
                         groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def relative_coordinate_feature_generator(self, mask_feat, instance_locations, strides):
        # obtain relative coordinate features for mask generator
        num_instance = len(instance_locations)
        H, W = mask_feat.size()[1:]
        locations = compute_locations(H,
                                      W,
                                      stride=8,
                                      device=mask_feat.device)
        relative_coordinates = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1) * 8.0)
        relative_coordinates = relative_coordinates.to(dtype=mask_feat.dtype)
        coordinates_feat = torch.cat([
            relative_coordinates.view(num_instance, 2, H, W),
            mask_feat.repeat(num_instance, 1, 1, 1)], dim=1)
        coordinates_feat = coordinates_feat.view(1, -1, H, W)
        return coordinates_feat

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
            top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   kernel_preds,
                   mask_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]

        mlvl_points, mlvl_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype,
            bbox_preds[0].device)

        det_results_list = []
        mask_results_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            kernel_pred_list = [
                kernel_preds[i][img_id].detach() for i in range(num_levels)
            ]
            mask_feats_i = mask_feats[img_id]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']
            det_bboxes, det_labels, det_masks = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                kernel_pred_list,
                mask_feats_i,
                mlvl_points,
                mlvl_strides,
                img_shape,
                scale_factor,
                ori_shape,
                cfg,
                rescale,
                with_nms)

            if det_bboxes.shape[0] == 0:
                det_results_list.append([np.zeros((0, 5), dtype=np.float32) for i in range(self.num_classes)])
                mask_results_list.append([np.zeros((0, 0), dtype=np.float32) for i in range(self.num_classes)])
                continue
            bbox_results = bbox2result(det_bboxes, det_labels, self.num_classes)

            mask_results = [[] for _ in range(self.num_classes)]
            for i in range(det_bboxes.shape[0]):
                label = det_labels[i]
                mask = det_masks[i].cpu().numpy()
                mask_results[label].append(mask)

            det_results_list.append(bbox_results)
            mask_results_list.append(mask_results)
        return det_results_list, mask_results_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           kernel_preds,
                           mask_feat,
                           mlvl_points,
                           mlvl_strides,
                           img_shape,
                           scale_factor,
                           ori_shape,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (1, H, W).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_kernels_pred = []
        flatten_mlvl_points = []
        flatten_mlvl_strides = []
        for cls_score, bbox_pred, centerness, kernel_pred, points, strides in zip(
                cls_scores, bbox_preds, centernesses, kernel_preds,  mlvl_points, mlvl_strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, 169)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds, :]
                strides = strides[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                kernel_pred = kernel_pred[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_kernels_pred.append(kernel_pred)
            flatten_mlvl_strides.append(strides)
            flatten_mlvl_points.append(points)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_kernels_pred = torch.cat(mlvl_kernels_pred)

        flatten_mlvl_points = torch.cat(flatten_mlvl_points)
        flatten_mlvl_strides = torch.cat(flatten_mlvl_strides)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels, det_kernels_pred, det_points, det_strides = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_kernels_pred,
            flatten_mlvl_points,
            flatten_mlvl_strides,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)

        # generate masks
        masks = []
        if det_bboxes.shape[0] > 0:
            mask_head_params = det_kernels_pred
            num_instance = len(det_points)
            mask_head_inputs = self.relative_coordinate_feature_generator(
                mask_feat,
                det_points,
                det_strides)
            weights, biases = parse_dynamic_params(
                mask_head_params,
                8,
                self.weight_nums,
                self.bias_nums)
            mask_logits = self.mask_heads_forward(
                mask_head_inputs,
                weights,
                biases,
                num_instance)
            mask_logits = mask_logits.reshape(-1, 1, mask_feat.size(1), mask_feat.size(2)).sigmoid()
            if rescale:
                pred_global_masks = aligned_bilinear(mask_logits, 8)
                pred_global_masks = pred_global_masks[:, :, :img_shape[0], :img_shape[1]]
                masks = F.interpolate(
                    pred_global_masks,
                    size=(ori_shape[0], ori_shape[1]),
                    mode='bilinear',
                    align_corners=False).squeeze(1)
            else:
                masks = aligned_bilinear(mask_logits, 8).squeeze(1)
            masks.gt_(0.5)
        return det_bboxes, det_labels, masks

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds_list = multi_apply(
            self.get_targets_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        return labels_list, bbox_targets_list, gt_inds_list

    def get_targets_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_zeros(num_points), gt_bboxes.new_zeros((num_points, 4)), \
                   gt_labels.new_zeros(num_points), gt_labels.new_zeros(num_points)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) \
            & (max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_ind = min_area_inds[labels < self.num_classes]

        return labels, bbox_targets, gt_ind

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single(
                featmap_sizes[i],
                self.strides[i],
                dtype,
                device)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        strides = points[:,0] * 0 + stride
        return points, strides
