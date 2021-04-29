import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, bias_init_with_prob, constant_init,
                      normal_init)
from mmcv.runner import force_fp32
from mmcv.ops.nms import batched_nms

from mmdet.core import (build_assigner, build_sampler,images_to_levels, multi_apply,  reduce_mean,
                        bbox_overlaps, multiclass_nms)
from .anchor_head import AnchorHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class TinaFaceHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 feat_channels=256,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_iou=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 detach=True,  # 新加
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.detach = detach
        super(TinaFaceHead, self).__init__(num_classes, in_channels, feat_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        #self.loss_centerness = build_loss(loss_iou)
        self.loss_iou = build_loss(loss_iou)

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
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_iou = nn.Conv2d(
            self.feat_channels, self.num_anchors, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        constant_init(self.retina_reg, 0.)
        constant_init(self.retina_iou, 0.)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        iou_pred = self.retina_iou(reg_feat)

        return cls_score, bbox_pred, iou_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   iou,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            iou (list[Tensor]): iou for each scale level with
                shape (N, num_anchors * 1, H, W).
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
            iou_pred_list = [
                iou[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                iou_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           ious,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            ious (list[Tensor]): iou for a single scale level
                with shape (num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
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
        alpha = cfg.get('alpha', None)
        height_th = cfg.get('height_th', None)
        min_scale = cfg.get('min_scale', 1)
        max_scale = cfg.get('max_scale', 10000)
        split_thr = cfg.get('split_thr', 10000)

        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, iou, anchors in zip(
                cls_scores, bbox_preds, ious, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            iou = iou.permute(1, 2, 0).reshape(-1, 1).sigmoid()

            if alpha is None:
                scores *= iou
            elif isinstance(alpha, float):
                scores = torch.pow(scores, 2 * alpha) * torch.pow(
                    iou, 2 * (1 - alpha))
            else:
                raise ValueError("alpha must be float or None")

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * iou[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)

        # TODO：Whether the function is valid
        keeps = self._filter_boxes(mlvl_bboxes, min_scale, max_scale)
        mlvl_bboxes = mlvl_bboxes[keeps]
        mlvl_scores = mlvl_scores[keeps]

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if height_th is not None:
            hs = mlvl_bboxes[:, 3] - mlvl_bboxes[:, 1]
            valid = (hs >= height_th)
            mlvl_bboxes, mlvl_scores = (
                mlvl_bboxes[valid], mlvl_scores[valid])

        if with_nms:
            det_bboxes, det_labels = self.split_batch_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                split_thr)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _filter_boxes(self, boxes, min_scale, max_scale):
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        scales = torch.sqrt(ws * hs)
        return (scales >= max(1, min_scale)) & (scales <= max_scale)

    def split_batch_nms(self,
                        multi_bboxes,
                        multi_scores,
                        score_thr,
                        nms_cfg,
                        max_num=-1,
                        split_thr=10000):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int): if there are more than max_num bboxes after NMS,
                only top max_num will be kept.

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
                are 0-based.
        """
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)

        scores = multi_scores[:, :-1]

        labels = torch.arange(num_classes, dtype=torch.long)
        labels = labels.view(1, -1).expand_as(scores)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        valid_mask = scores > score_thr
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

        if bboxes.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            return bboxes, labels

        inds = scores.argsort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

        batch_bboxes = torch.empty((0, 4),
                                   dtype=bboxes.dtype,
                                   device=bboxes.device)
        batch_scores = torch.empty((0,), dtype=scores.dtype, device=scores.device)
        batch_labels = torch.empty((0,), dtype=labels.dtype, device=labels.device)
        while bboxes.shape[0] > 0:
            num = min(int(split_thr), bboxes.shape[0])
            batch_bboxes = torch.cat([batch_bboxes, bboxes[:num]])
            batch_scores = torch.cat([batch_scores, scores[:num]])
            batch_labels = torch.cat([batch_labels, labels[:num]])
            bboxes = bboxes[num:]
            scores = scores[num:]
            labels = labels[num:]

            _, keep = batched_nms(batch_bboxes, batch_scores, batch_labels,
                                  nms_cfg)
            batch_bboxes = batch_bboxes[keep]
            batch_scores = batch_scores[keep]
            batch_labels = batch_labels[keep]

        dets = torch.cat([batch_bboxes, batch_scores[:, None]], dim=-1)
        labels = batch_labels

        if max_num > 0:
            dets = dets[:max_num]
            labels = labels[:max_num]

        return dets, labels

    def loss_single(self, cls_score, bbox_pred, iou_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):  # 区别
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        anchors = anchors.reshape(-1, 4)  # 新加
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

        iou_targets = label_weights.new_zeros(labels.shape)  # 新加
        iou_weights = label_weights.new_zeros(labels.shape)  # 新加
        iou_weights[(bbox_weights.sum(axis=1) > 0).nonzero(as_tuple=False)] = 1.  # 新加
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)  # 新加

        bg_class_ind = self.num_classes  # 新加
        pos_inds = ((labels >= 0) &
                    (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)  # 新加

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        if len(pos_inds) > 0:
            # dx, dy, dw, dh
            pos_bbox_targets = bbox_targets[pos_inds]
            # tx, ty, tw, th
            pos_bbox_pred = bbox_pred[pos_inds]
            # x1, y1, x2, y2
            pos_anchors = anchors[pos_inds]

            if self.reg_decoded_bbox:
                pos_decode_bbox_pred = pos_bbox_pred
                gt_bboxes = pos_bbox_targets
            else:
                # x1, y1, x2 ,y2
                pos_decode_bbox_pred = self.bbox_coder.decode(
                    pos_anchors, pos_bbox_pred)

                gt_bboxes = self.bbox_coder.decode(pos_anchors,
                                                   pos_bbox_targets)

            if self.detach:
                pos_decode_bbox_pred = pos_decode_bbox_pred.detach()

            iou_targets[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred, gt_bboxes, is_aligned=True)

        loss_iou = self.loss_iou(
            iou_pred, iou_targets, iou_weights, avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_iou

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,  # 新加
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
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
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # num_total_samples = (num_total_pos + num_total_neg if self.sampling else num_total_pos)
        num_total_samples = reduce_mean(
            torch.tensor(1. * num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)  # 区别

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_iou = multi_apply(  # 区别
            self.loss_single,
            cls_scores,
            bbox_preds,
            iou_preds,  # 区别
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, losses_iou=losses_iou)
