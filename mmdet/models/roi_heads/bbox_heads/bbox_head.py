# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class BBoxHead(BaseModule):
    """最简单的 RoI Head,只有两个 fc 层分别用于分类和回归."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg=None):
        super(BBoxHead, self).__init__(init_cfg)
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fp16_enabled = False

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area  # 256 * 7 * 7
        if self.with_cls:
            # need to add background class
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=in_channels,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=in_channels,
                out_features=out_dim_reg)
        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """根据采样结果计算单张图片中正负样本对应的cls target与reg target.

        Args:
            pos_bboxes (Tensor): [num_pos, 4], batch幅图像的正样本
            neg_bboxes (Tensor): [num_neg, 4], batch幅图像的负样本
            pos_gt_bboxes (Tensor): [num_pos, 4], 正样本对应的的gt box
            pos_gt_labels (Tensor): [num_pos, ], 正样本对应的的gt label
            cfg (obj:`ConfigDict`): R-CNN的训练配置,仅使用pos_weight作为正样本的cls weight

        Returns:
            Tuple[Tensor]: 单张图片中正负样本的拟合目标:

                - labels(Tensor): cls目标, [num_pos+num_neg, ].
                - label_weights(Tensor): cls权重, [num_pos+num_neg, ].
                - bbox_targets(Tensor): reg目标, [num_pos+num_neg, 4].
                - bbox_weights(Tensor): reg权重, [num_pos+num_neg, 4].
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            # reg_decoded_bbox为True(loss为IOU类时).此时对绝对坐标计算loss
            # 否则对anchor/roi/point -> gt box 的修正系数计算loss
            if self.reg_decoded_bbox:
                pos_bbox_targets = pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """根据sampling_results计算一个batch中所有样本对应的gt box.

        与 bbox_head 中的实现几乎相同, 我们将附加参数 pos_inds_list 和
        neg_inds_list 传递给 `_get_target_single` 函数.

        Args:
            sampling_results (List[obj:SamplingResults]): batch幅图像的样本采样结果.
            gt_bboxes (list[Tensor]): [[num_gt, 4],] * bs
            gt_labels (list[Tensor]): [[num_gt, ],] * bs
            rcnn_train_cfg (obj:ConfigDict): r-cnn的训练配置.
            concat (bool): 是否将batch张图像的结果cat到一起.

        Returns:
            Tuple[Tensor]: roi对应的cls target与reg target.
            包含以下张量:

                - labels (list[Tensor],Tensor): [num_roi, ] * bs
                  如果 concat为True时,则为[bs * num_roi, ]. 下同
                - label_weights (list[Tensor]): [num_roi, ] * bs
                - bbox_targets (list[Tensor],Tensor): [num_roi, 4] * bs
                - bbox_weights (list[tensor],Tensor): [num_roi, 4] * bs
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        """
        Args:
            cls_score (Tensor): 根据正负样本输出的cls score, [bs*(pos_num+neg_num), nc]
            bbox_pred (Tensor): 根据正负样本输出的reg score, [bs*(pos_num+neg_num), nc*4]
            rois (Tensor): rpn或r-cnn提供的roi,然后根据与gt box匹配与采样后的结果提取出
                [bs*(pos_num+neg_num), 5] 5 -> [img_index, x1, y1, x2, y2]
            labels (Tensor): rois对应的gt label[bs*(pos_num+neg_num), ]
            label_weights (Tensor): rois对应的cls weight[bs*(pos_num+neg_num), ]
            bbox_targets (Tensor): rois对应的gt box[bs*(pos_num+neg_num), 4]
            bbox_weights (Tensor): rois对应的reg weight[bs*(pos_num+neg_num), 4]
            reduction_override (bool, str): 代表是否覆盖Loss类初始化中的reduction
        """
        losses = dict()
        if cls_score is not None:  # avg_factor为batch幅图像中正样本总和
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,  # 该参数仅在最终reduction=mean时才生效
                    reduction_override=reduction_override)
                # 一般来说是 torch.Tensor类型的
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 前景:[0, nc), 背景:nc
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # "基础box"在不同的地方寓意不同,在一阶段(或者二阶段的rpn中)
                    # 指代anchor(如果是anchor base的网络的话),其余地方代指 roi
                    # 该参数为True时,意为需要根据"基础box"与网络预测的修正系数结合.
                    # 计算得到box的绝对坐标与gt box一同送入损失函数进行计算loss
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
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
        """将网络的单张图片输出转换为最终检测box与label.

        Args:
            rois (Tensor): 将要进行修正的box. (num_boxes, 5).
               5 -> (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, (num_boxes, nc + 1).
            bbox_pred (Tensor, optional): 网络回归值.(num_boxes, 4).
            img_shape (Sequence[int], optional): 框的最大界限,
               指定 (H, W, C) or (H, W).
            scale_factor (ndarray): 图像的缩放因子,参考transforms.py中的Resize类
                其格式为(w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, 对box进行逆放缩,以匹配其在原始图像的位置.
             cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:`det_bboxes`: (num_boxes, 5)
                5 -> (x1, y1, x2, y2, score).
                `det_bboxes`: (num_boxes, ).
        """

        # 一些loss,比如(Seesaw loss..) 可能有自定义激活函数
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # 当 with_reg 为 False 时,bbox_pred 在某些检测器中将为 None,
        # 比如 Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """在训练期间微调box. 令N=bs*(pos_num+neg_num)

        Args:
            rois (Tensor): [N, 5], [img_ind, x1, y1, x2, y2]
            labels (Tensor): [N, ].
            bbox_preds (Tensor): [N, 4] or [N, 4*#class].
            pos_is_gts (list[Tensor]): [[num_pos, ], ] * bs. roi是否为gt的mask.
            img_metas (list[dict]): [dict(), ] * bs, 每个图像的元信息.

        Returns:
            list[Tensor]: [[N-num_gt, 5], ] * bs, 调整后的box,同时也过滤了gt.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze(dim=1)
            num_rois = inds.numel()  # 当前图像的正负样本数量

            # 归属于某张图片的 roi [num_pos+num_neg, 4]
            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # 过滤掉box中的gt
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """根据roi类别获取指定回归结果来修正roi. Used in Cascade R-CNN.

        Args:
            rois (Tensor): 来自`rpn_head` 或 `roi head的bbox_head`的pred_box,
                [sample.num, 4/5].
            label (Tensor): 仅在reg_class_agnostic为False时使用, [sample.num, ].
            bbox_pred (Tensor): box head的reg输出,对roi再次修正 reg_class_agnostic
                为True时,表示当前box仅仅是前景[sample.num, 4],仅有一个(前景).
                为False时,表示当前box在不同类别上独立回归[sample.num, nc * 4],所以就根据
                最大的cls score所对应的cls ind(上面的label参数)来获取相应的reg score.
            img_meta (dict): 图像元信息.

        Returns:
            Tensor: 微调后的roi, [sample.num, 4/5].
        """

        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        max_shape = img_meta['img_shape']

        if rois.size(1) == 4:
            new_rois = self.bbox_coder.decode(
                rois, bbox_pred, max_shape=max_shape)
        else:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=max_shape)
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)

        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(
                bboxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets

            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
