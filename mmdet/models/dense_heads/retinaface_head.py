# Copyright (c) OpenMMLab. All rights reserved.
from inspect import signature

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class RetinaFaceHead(AnchorHead):
    r"""An anchor-based head used in `RetinaFace
    <https://arxiv.org/pdf/1905.00641v2.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaFaceHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=0,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaFaceHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.sampling = False

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
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]

        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list

    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.

        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
            mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1),
            device=mlvl_cls_scores[0].device,
            dtype=torch.long)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(0, 2, 3,
                                          1).reshape(batch_size, -1,
                                                     self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            anchors = anchors.expand_as(bbox_pred)
            # Always keep topk op for dynamic input in onnx
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[..., :-1].max(-1)

                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds)
                anchors = anchors[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx

            # ignore background class
            if not self.use_sigmoid_cls:
                num_classes = batch_mlvl_scores.shape[2] - 1
                batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = batch_mlvl_scores.new_zeros(batch_size,
                                                  batch_mlvl_scores.shape[1],
                                                  1)
            batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
                                                  batch_mlvl_scores):
                # use split_batch_nms instead of multiclass_nms to avoid
                # GPU OOM when dealing with images with large number of dets
                det_bbox, det_label = self.split_batch_nms(
                    mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                    cfg.max_per_img)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
            ]
        return det_results

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        if hasattr(self, '_get_bboxes'):
            gbs_sig = signature(self._get_bboxes)
        else:
            gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None

        # use batched nms instead in multi-scale testing
        det_bboxes, det_labels = self.split_batch_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def split_batch_nms(self,
                        multi_bboxes,
                        multi_scores,
                        score_thr,
                        nms_cfg,
                        max_num=-1,
                        score_factors=None,
                        split_thr=30000):
        """Batched NMS for multi-class bboxes.
        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class.
            score_thr (float): bbox threshold, bboxes with scores lower
                will not be considered.
            nms_cfg (dict): Config dict for NMS.
            max_num (int, optional): if there are more than max_num bboxes
                after NMS, only top max_num will be kept. Default to -1.
            score_factors (Tensor, optional): The factors multiplied to
                scores before applying NMS. Default to None.
            split_thr (int, optional): Max number of detections in one batch.
                Default to 30000.

        Returns:
            tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1).
                Labels are 0-based.
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

        # multiply score_factor after threshold to preserve more bboxes,
        # improve mAP by 1% for YOLOv3
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

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
        batch_scores = torch.empty((0, ),
                                   dtype=scores.dtype,
                                   device=scores.device)
        batch_labels = torch.empty((0, ),
                                   dtype=labels.dtype,
                                   device=labels.device)
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

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)  # [12800 * 8, ]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)  # [8 * 80 * 80 * 2, 2]
        loss_cls_all = self.loss_cls(cls_score, labels,
                                     label_weights).reshape(-1)

        # FG cat_id: 0, BG cat_id: 1
        pos_inds = (labels == 0).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == 1).nonzero(as_tuple=False).reshape(
            -1)  # .view(-1)

        # OHEM
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()

        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        return loss_cls[None], loss_bbox
