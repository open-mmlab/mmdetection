# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import cv2
import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, force_fp32,
                        images_to_levels, multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class YOLOV3Head(BaseDenseHead):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth.
            conventionally this value is set to 0.05
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=None,
                 loss_conf=None,
                 loss_xy=None,
                 loss_wh=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV3Head, self).__init__()
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))
        assert (loss_cls is not None) and (loss_conf is not None)
        assert (loss_bbox is not None) ^ (
            loss_xy is not None and loss_wh is not None)

        self.using_iou_loss = loss_bbox is not None
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        if self.using_iou_loss:
            self.loss_bbox = build_loss(loss_bbox)
        else:
            self.loss_xy = build_loss(loss_xy)
            self.loss_wh = build_loss(loss_wh)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModule(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self, pred_maps, img_metas, cfg=None, rescale=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if multi_lvl_conf_scores.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0],
                                                 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding],
                                         dim=1)

        det_bboxes, det_labels = multiclass_nms(
            multi_lvl_bboxes,
            multi_lvl_cls_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=multi_lvl_conf_scores)

        return det_bboxes, det_labels

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (num_anchors, N, feat_dim, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device

        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.anchor_generator.responsible_flags(
                    featmap_sizes, gt_bboxes[img_id], device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        # losses_cls = torch.Tensor(.0, device=device)
        # losses_conf = torch.Tensor(.0, device=device)
        # losses_xy = torch.Tensor(.0, device=device)
        # losses_wh = torch.Tensor(.0, device=device)
        # for i in range(self.num_levels):
        #     d_losses_cls, d_losses_conf, d_losses_xy, d_losses_wh =
        #     multi_apply(
        #         self.loss_single, pred_maps[i], target_maps_list[i],
        #         neg_maps_list[i],
        #         multi_level_anchors[i])
        #     losses_cls += d_losses_cls.sum()
        #     losses_conf += d_losses_conf.sum()
        #     losses_xy += d_losses_xy.sum()
        #     losses_wh += d_losses_wh.sum()

        level_idx_list = list(range(self.num_levels))
        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single_level, pred_maps, target_maps_list, neg_maps_list,
            multi_level_anchors, level_idx_list)

        return dict(
            loss_cls=losses_cls,
            loss_conf=losses_conf,
            loss_xy=losses_xy,
            loss_wh=losses_wh)

    def loss_single_level(self, pred_map, target_map, neg_map, anchors,
                          level_idx):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level. Dimension:
                [batch_size, feat_dim, feat_w, feat_h]
            target_map (Tensor): The Ground-Truth target for a single level.
                Dimension: [batch_size, num_anchors, 85]
            neg_map (Tensor): The negative masks for a single level.
                Dimension: [batch_size, num_anchors]
            anchors (Tensor): The anchors for a single level.
                Dimension: [num_anchors]


        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        device = pred_map.device
        num_cls = self.num_attrib - 5
        assert num_cls > 0

        # print(pred_map.shape)

        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask
        pos_mask = pos_mask.unsqueeze(dim=-1)
        # pos_and_neg_mask = pos_and_neg_mask.unsqueeze(dim=-1)
        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        loss_cls = torch.tensor(.0, device=device, requires_grad=True)
        loss_conf = torch.tensor(.0, device=device, requires_grad=True)
        loss_xy = torch.tensor(.0, device=device, requires_grad=True)
        loss_wh = torch.tensor(.0, device=device, requires_grad=True)

        for i in range(num_imgs):

            img_pos_mask = pos_mask[i, :, :]
            if img_pos_mask.sum() <= 0:
                continue

            img_pred_conf = pred_map[i, :, 4]
            img_pred_label = pred_map[i, :, 5:]
            img_target_conf = target_map[i, :, 4]
            img_target_label = target_map[i, :, 5:]

            img_cls_mask = img_pos_mask.expand(-1, num_cls).bool()
            img_pred_label = img_pred_label.masked_select(
                img_cls_mask).reshape(-1, num_cls)
            img_target_label = img_target_label.masked_select(
                img_cls_mask).reshape(-1, num_cls)

            loss_cls = loss_cls + self.loss_cls(img_pred_label,
                                                img_target_label)
            loss_conf = loss_conf + self.loss_conf(img_pred_conf,
                                                   img_target_conf)
            #
            # print(img_target_conf.shape)
            # print(img_target_label.shape)
            # print(loss_cls)

            if self.using_iou_loss:
                # preparation for box decoding
                anchor_strides = torch.tensor(
                    self.featmap_strides[level_idx],
                    device=pred_map.device).repeat(len(anchors))
                assert len(anchor_strides) == len(anchors)
                img_pred_xywh = pred_map[i, :, :4].reshape(-1, 4).contiguous()
                # decode box for IoU loss
                img_pred_box = self.bbox_coder.decode(anchors, img_pred_xywh,
                                                      anchor_strides)

                img_target_box = \
                    target_map[i, :, :4].reshape(-1, 4).contiguous()
                # img_target_box = self.bbox_coder.decode(
                #     anchors, img_target_box, anchor_strides)

                img_box_mask = \
                    img_pos_mask.reshape(-1, 1).contiguous().expand(-1,
                                                                    4).bool()
                img_pred_box = \
                    img_pred_box.masked_select(img_box_mask).reshape(-1, 4)
                img_target_box = \
                    img_target_box.masked_select(img_box_mask).reshape(-1, 4)

                # img_meta = data_batch['img_metas'].data[0]
                # img = data_batch['img'].data[0].squeeze().detach().cpu(
                # ).numpy()
                # img = img.transpose(1, 2, 0)[:,:,::-1]
                # # img = img.clip(0.01, 0.99)
                # # import numpy as np
                # # img = np.float64(img)
                # # img = img.copy()

                # if level_idx == 0:
                #     self.show_img(img_pred_box, img_target_box)

                loss_xy = loss_xy + self.loss_bbox(img_pred_box,
                                                   img_target_box)
                loss_wh = loss_wh + torch.zeros_like(loss_xy)
            else:
                img_pred_xy = pred_map[i, :, :2]
                img_pred_wh = pred_map[i, :, 2:4]
                img_target_xy = target_map[i, :, :2]
                img_target_wh = target_map[i, :, 2:4]
                loss_xy = loss_xy + self.loss_xy(
                    img_pred_xy, img_target_xy, weight=img_pos_mask)
                loss_wh = loss_wh + self.loss_wh(
                    img_pred_wh, img_target_wh, weight=img_pos_mask)

        return loss_cls, loss_conf, loss_xy, loss_wh

        # anchors = # need batch_size set of anchors
        # img_index_list = list(range(num_imgs))
        # all_losses_per_image = multi_apply(
        #     self.loss_single_img, pred_map, target_map, neg_map,
        #     anchors, level_idx_list, img_index_list)
        #
        # all_losses = [torch.stack(item, dim=0).sum() for item in
        #             all_losses_per_image]
        # # loss_cls, loss_conf, loss_xy, loss_wh = all_losses
        #
        # return all_losses  # loss_cls, loss_conf, loss_xy, loss_wh

    # def loss_single_img(self, pred_map, target_map, neg_map, anchors,
    #                       level_idx):

    def show_img(self, img_pred_box, img_target_box=None, center=False):

        def center_box(boxes):
            import numpy as np
            xy = boxes[:, 0:2] + 0.5 * (boxes[:, 2:4] - boxes[:, 0:2])
            wh = np.ones_like(xy) * 5
            return np.hstack((xy - wh, xy + wh))

        img = cv2.imread('data/coco/val2017/000000182611.jpg')
        img = cv2.resize(img, (480, 640))
        if not center:
            bbox = img_pred_box.detach().cpu().numpy()
            mmcv.imshow_bboxes(img[:], bbox)
            if img_target_box is not None:
                bbox2 = img_target_box.detach().cpu().numpy()
                mmcv.imshow_bboxes(img[:], bbox2, 'red')
        else:
            bbox = center_box(img_pred_box.detach().cpu().numpy())
            mmcv.imshow_bboxes(img[:], bbox)
            if img_target_box is not None:
                bbox2 = center_box(img_target_box.detach().cpu().numpy())
                mmcv.imshow_bboxes(img[:], bbox2, 'red')

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (Num_total_gt, num_total_anchors)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        results = multi_apply(self._get_targets_single, anchor_list,
                              responsible_flag_list, gt_bboxes_list,
                              gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        anchor_strides = []
        for i in range(len(anchors)):
            anchor_strides.append(
                torch.tensor(self.featmap_strides[i],
                             device=gt_bboxes.device).repeat(len(anchors[i])))
        concat_anchors = torch.cat(anchors)
        concat_responsible_flags = torch.cat(responsible_flags, dim=1)

        anchor_strides = torch.cat(anchor_strides)
        assert len(anchor_strides) == len(concat_anchors) == \
               concat_responsible_flags.shape[1]
        assign_result = self.assigner.assign(concat_anchors,
                                             concat_responsible_flags,
                                             gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors,
                                              gt_bboxes)

        target_map = concat_anchors.new_zeros(
            concat_anchors.size(0), self.num_attrib)

        # for i in range(sampling_result.pos_bboxes.shape[0]):
        #     self.show_img(
        #         sampling_result.pos_bboxes[i:i+1, :],
        #         sampling_result.pos_gt_bboxes[i:i+1, :],
        #         center=False)

        if self.using_iou_loss:
            target_map[sampling_result.pos_inds, :4] = \
                sampling_result.pos_gt_bboxes
        else:
            target_map[sampling_result.pos_inds, :4] = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes,
                anchor_strides[sampling_result.pos_inds])

        target_map[sampling_result.pos_inds, 4] = 1

        gt_labels_one_hot = F.one_hot(
            gt_labels, num_classes=self.num_classes).float()
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (
                1 - self.one_hot_smoother
            ) + self.one_hot_smoother / self.num_classes
        target_map[sampling_result.pos_inds, 5:] = gt_labels_one_hot[
            sampling_result.pos_assigned_gt_inds]

        neg_map = concat_anchors.new_zeros(
            concat_anchors.size(0), dtype=torch.uint8)
        neg_map[sampling_result.neg_inds] = 1

        return target_map, neg_map
