import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import (build_sampler, fast_nms, force_fp32, images_to_levels,
                        multi_apply)
from ..builder import HEADS, build_loss
from ..losses import smooth_l1_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class YolactHead(AnchorHead):
    """An anchor-based head used in YOLACT [1]_.

    References:
        .. [1]  https://arxiv.org/pdf/1904.02689.pdf
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_head_convs=1,
                 num_protos=32,
                 use_OHEM=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=1.5),
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=3,
                     scales_per_octave=1,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_OHEM = use_OHEM
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(YolactHead, self).__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            **kwargs)
        if self.use_OHEM:
            self.loss_cls_weight = loss_cls.get('loss_weight', 1.0)
            self.loss_bbox_weight = loss_bbox.get('loss_weight', 1.0)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampling = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = nn.ModuleList()
        for i in range(self.num_head_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.head_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.yolact_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.yolact_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.yolact_coeff = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.num_protos,
            3,
            padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.head_convs:
            xavier_init(m.conv, distribution='uniform', bias=0)
        xavier_init(self.yolact_cls, distribution='uniform', bias=0)
        xavier_init(self.yolact_reg, distribution='uniform', bias=0)
        xavier_init(self.yolact_coeff, distribution='uniform', bias=0)

    def forward_single(self, x):
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.yolact_cls(x)
        bbox_pred = self.yolact_reg(x)
        coeff_pred = self.yolact_coeff(x).tanh()
        return cls_score, bbox_pred, coeff_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=not self.use_OHEM,
            return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results) = cls_reg_targets

        if self.use_OHEM:
            num_images = len(img_metas)
            all_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, self.cls_out_channels) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)
            all_label_weights = torch.cat(label_weights_list,
                                          -1).view(num_images, -1)
            all_bbox_preds = torch.cat([
                b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
                for b in bbox_preds
            ], -2)
            all_bbox_targets = torch.cat(bbox_targets_list,
                                         -2).view(num_images, -1, 4)
            all_bbox_weights = torch.cat(bbox_weights_list,
                                         -2).view(num_images, -1, 4)

            # concat all level anchors to a single tensor
            all_anchors = []
            for i in range(num_images):
                all_anchors.append(torch.cat(anchor_list[i]))

            # check NaN and Inf
            assert torch.isfinite(all_cls_scores).all().item(), \
                'classification scores become infinite or NaN!'
            assert torch.isfinite(all_bbox_preds).all().item(), \
                'bbox predications become infinite or NaN!'

            losses_cls, losses_bbox = multi_apply(
                self.loss_single_OHEM,
                all_cls_scores,
                all_bbox_preds,
                all_anchors,
                all_labels,
                all_label_weights,
                all_bbox_targets,
                all_bbox_weights,
                num_total_samples=num_total_pos)
        else:
            num_total_samples = (
                num_total_pos +
                num_total_neg if self.sampling else num_total_pos)

            # anchor number of multi levels
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # concat all level anchors and flags to a single tensor
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                               num_level_anchors)
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,
                bbox_preds,
                all_anchor_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_samples=num_total_samples)

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox), sampling_results

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def loss_single_OHEM(self, cls_score, bbox_pred, anchors, labels,
                         label_weights, bbox_targets, bbox_weights,
                         num_total_samples):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) &
                    (labels < self.background_label)).nonzero().reshape(-1)
        neg_inds = (labels == self.background_label).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos +
                    loss_cls_neg) / num_total_samples * self.loss_cls_weight
        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg.smoothl1_beta,
            avg_factor=num_total_samples) * self.loss_bbox_weight
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'coeff_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   coeff_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            coeff_pred_list = [
                coeff_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                coeff_pred_list, mlvl_anchors,
                                                img_shape, scale_factor, cfg,
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           coeff_preds_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, anchors in \
                zip(cls_score_list, bbox_pred_list,
                    coeff_preds_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            coeff_pred = coeff_pred.permute(1, 2,
                                            0).reshape(-1, self.num_protos)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_coeffs = torch.cat(mlvl_coeffs)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores,
                                                      mlvl_coeffs,
                                                      cfg.score_thr, cfg.nms,
                                                      cfg.max_per_img)
        return det_bboxes, det_labels, det_coeffs


@HEADS.register_module()
class YolactSegmHead(nn.Module):

    def __init__(self,
                 in_channels=256,
                 num_classes=80,
                 loss_segm=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0)):
        super(YolactSegmHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_segm = build_loss(loss_segm)
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def init_weights(self):
        xavier_init(self.segm_conv, distribution='uniform')

    def forward(self, x):
        return self.segm_conv(x)

    @force_fp32(apply_to=('segm_pred', ))
    def loss(self, segm_pred, gt_masks, gt_labels):
        loss_segm = []
        num_imgs, num_classes, mask_h, mask_w = segm_pred.size()
        for idx in range(num_imgs):
            cur_segm_pred = segm_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_labels = gt_labels[idx]

            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    cur_gt_masks.unsqueeze(0), (mask_h, mask_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segm_targets = torch.zeros_like(
                    cur_segm_pred, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segm_targets[cur_gt_labels[obj_idx] - 1] = torch.max(
                        segm_targets[cur_gt_labels[obj_idx] - 1],
                        downsampled_masks[obj_idx])

            loss = self.loss_segm(
                cur_segm_pred,
                segm_targets,
                avg_factor=num_imgs * mask_h * mask_w)
            loss_segm.append(loss)
        return dict(loss_segm=loss_segm)


@HEADS.register_module()
class YolactProtonet(nn.Module):
    """Protonet of Yolact.

    This head outputs the prototypes for Yolact.
    """

    def __init__(
        self,
        in_channels=256,
        num_protos=32,
        num_classes=80,
        loss_mask_weight=1.0,
        max_masks_to_train=100,
    ):
        super(YolactProtonet, self).__init__()
        config = [(256, 3, {'padding': 1})] * 3 + \
            [(None, -2, {}), (256, 3, {'padding': 1})] \
            + [(num_protos, 1, {})]
        self.protonet = self.make_net(in_channels, config)
        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.fp16_enabled = False

    def make_net(self, in_channels, config, include_last_relu=True):
        """A helper function to take a config setting and turn it into a
        network.

        Used by protonet and extrahead. Returns (network)
        """

        def make_layer(layer_cfg):
            nonlocal in_channels

            # Possible patterns:
            # ( 256, 3, {}) -> conv
            # ( 256,-2, {}) -> deconv
            # (None,-2, {}) -> bilinear interpolate

            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size,
                                  **layer_cfg[2])
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False,
                        **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels,
                                               -kernel_size, **layer_cfg[2])

            in_channels = num_channels if num_channels is not None \
                else in_channels

            return [layer, nn.ReLU(inplace=True)]

        # Use sum to concat together all the component layer lists
        net = sum([make_layer(x) for x in config], [])
        if not include_last_relu:
            net = net[:-1]

        return nn.Sequential(*(net))

    def init_weights(self):
        for m in self.protonet:
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x, coeff_pred, bboxes, img_meta, sampling_results=None):
        prototypes = self.protonet(x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = x.size(0)
        # Training state
        if sampling_results is not None:
            coeff_pred_list = []
            for coeff_pred_per_level in coeff_pred:
                coeff_pred_per_level = \
                    coeff_pred_per_level.permute(0, 2, 3, 1)\
                    .reshape(num_imgs, -1, self.num_protos)
                coeff_pred_list.append(coeff_pred_per_level)
            coeff_pred = torch.cat(coeff_pred_list, dim=1)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            cur_coeff_pred = coeff_pred[idx]
            cur_bboxes = bboxes[idx]
            cur_img_meta = img_meta[idx]

            # Testing state
            if sampling_results is None:
                bboxes_for_cropping = cur_bboxes
            else:
                cur_sampling_results = sampling_results[idx]
                pos_assigned_gt_inds = \
                    cur_sampling_results.pos_assigned_gt_inds
                bboxes_for_cropping = cur_bboxes[pos_assigned_gt_inds].clone()
                pos_inds = cur_sampling_results.pos_inds
                cur_coeff_pred = cur_coeff_pred[pos_inds]

            mask_pred = cur_prototypes @ cur_coeff_pred.t()
            mask_pred = torch.sigmoid(mask_pred)

            h, w = cur_img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0] /= w
            bboxes_for_cropping[:, 1] /= h
            bboxes_for_cropping[:, 2] /= w
            bboxes_for_cropping[:, 3] /= h

            mask_pred = self.crop(mask_pred, bboxes_for_cropping)
            mask_pred = mask_pred.permute(2, 0, 1).contiguous()
            mask_pred_list.append(mask_pred)
        return mask_pred_list

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, gt_bboxes, img_meta, sampling_results):
        loss_mask = []
        num_imgs = len(mask_pred)
        total_pos = 0
        for idx in range(num_imgs):
            cur_mask_pred = mask_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_bboxes = gt_bboxes[idx]
            cur_img_meta = img_meta[idx]
            cur_sampling_results = sampling_results[idx]

            pos_assigned_gt_inds = cur_sampling_results.pos_assigned_gt_inds
            num_pos = pos_assigned_gt_inds.size(0)
            if num_pos * cur_gt_masks.size(0) == 0:
                continue
            if num_pos > self.max_masks_to_train:
                perm = torch.randperm(num_pos)
                select = perm[:self.max_masks_to_train]
                cur_mask_pred = cur_mask_pred[select]
                pos_assigned_gt_inds = pos_assigned_gt_inds[select]
                num_pos = self.max_masks_to_train
            total_pos += num_pos

            gt_bboxes_for_reweight = cur_gt_bboxes[pos_assigned_gt_inds]

            mask_h, mask_w = cur_mask_pred.shape[-2:]
            cur_gt_masks = F.interpolate(
                cur_gt_masks.unsqueeze(0), (mask_h, mask_w),
                mode='bilinear',
                align_corners=False).squeeze(0)
            cur_gt_masks = cur_gt_masks.gt(0.5).float()
            mask_targets = cur_gt_masks[pos_assigned_gt_inds]

            cur_mask_pred = torch.clamp(cur_mask_pred, 0, 1)
            loss = F.binary_cross_entropy(
                cur_mask_pred, mask_targets,
                reduction='none') * self.loss_mask_weight

            h, w = cur_img_meta['img_shape'][:2]
            gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                               gt_bboxes_for_reweight[:, 0]) / w
            gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                gt_bboxes_for_reweight[:, 1]) / h
            loss = loss.mean(dim=(1, 2)) / gt_bboxes_width / gt_bboxes_height
            loss = torch.sum(loss)
            loss_mask.append(loss)

        loss_mask = [x / total_pos for x in loss_mask]

        return dict(loss_mask=loss_mask)

    def get_seg_masks(self, mask_pred, label_pred, img_meta, rescale):
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor[1]).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor[0]).astype(np.int32)
            scale_factor = 1.0

        mask_pred = mask_pred.cpu().numpy()
        mask_pred = mask_pred.astype(np.float32)
        cls_segms = [[] for _ in range(self.num_classes)]

        for m, l in zip(mask_pred, label_pred):
            im_mask = mmcv.imresize(m, (img_w, img_h))
            im_mask = (im_mask > 0.5).astype(np.uint8)
            cls_segms[l].append(im_mask)
        return cls_segms

    def crop(self, masks, boxes, padding=1):
        """"Crop" predicted masks by zeroing out everything not in the
        predicted bbox.

        Args:
            masks should be a size [h, w, n] tensor of masks
            boxes should be a size [n, 4] tensor of bbox coords in
                relative point form
        """
        h, w, n = masks.size()
        x1, x2 = self.sanitize_coordinates(
            boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(
            boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = torch.arange(
            w, device=masks.device, dtype=x1.dtype).view(1, -1,
                                                         1).expand(h, w, n)
        cols = torch.arange(
            h, device=masks.device, dtype=x1.dtype).view(-1, 1,
                                                         1).expand(h, w, n)

        masks_left = rows >= x1.view(1, 1, -1)
        masks_right = rows < x2.view(1, 1, -1)
        masks_up = cols >= y1.view(1, 1, -1)
        masks_down = cols < y2.view(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask.float()

    def sanitize_coordinates(self, _x1, _x2, img_size, padding=0, cast=True):
        """Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
        and x2 <= image_size. Also converts from relative to absolute
        coordinates and casts the results to long tensors.

        If cast is false, the result won't be cast to longs.
        Warning: this does things in-place behind the scenes so
        copy if necessary.
        """
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size
        if cast:
            _x1 = _x1.long()
            _x2 = _x2.long()
        x1 = torch.min(_x1, _x2)
        x2 = torch.max(_x1, _x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)

        return x1, x2


class InterpolateModule(nn.Module):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)
