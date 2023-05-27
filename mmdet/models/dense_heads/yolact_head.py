# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, force_fp32

from mmdet.core import build_sampler, fast_nms, images_to_levels, multi_apply
from mmdet.core.utils import select_single_mlvl
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead


@HEADS.register_module()
class YOLACTHead(AnchorHead):
    """YOLACT box head used in https://arxiv.org/abs/1904.02689.

    注意, YOLACT head 是 RetinaNet head的轻量版本.
    四个区别描述如下:

    1. YOLACT box head 在单个特征点上生成的anchor是RetinaNet的三分之一.
    2. YOLACT box head 共享 box 和 cls 分支的卷积.
    3. YOLACT box head 使用 OHEM 代替 Focal loss.
    4. YOLACT box head 为每个box预测一组mask coefficients.

    Args:
        num_classes (int): 分割类别数,不包含背景.
        in_channels (int): 输入特征图的维度.
        anchor_generator (dict): anchor生成策略的配置文件
        loss_cls (dict): cls loss的配置文件.
        loss_bbox (dict): reg loss的配置文件.
        num_head_convs (int): box 和 cls 分支共享的卷积层数.
        num_protos (int): mask coefficients的数量.
        use_ohem (bool): 如果为true, 则使用``loss_single_OHEM``计算cls loss
            否则使用``loss_single``.
        conv_cfg (dict): 构建和配置conv层的字典.
        norm_cfg (dict): 构建和配置norm层的字典.
        init_cfg (dict or list[dict], optional): 模型权重初始化的配置字典.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=3,
                     scales_per_octave=1,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='none',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
                 num_head_convs=1,
                 num_protos=32,
                 use_ohem=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     bias=0,
                     layer='Conv2d'),
                 **kwargs):
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_ohem = use_ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(YOLACTHead, self).__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        if self.use_ohem:
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampling = False

    def _init_layers(self):
        """Initialize layers of the head.
            conv3x3 * num_head_conv + conv_reg/cls/mask/
        """
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = ModuleList()
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
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
        self.conv_coeff = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.num_protos,
            3,
            padding=1)

    def forward_single(self, x):
        """单层级上的前向传播.

        Args:
            x (Tensor): 单层级特征图.

        Returns:
            tuple:
                cls_score (Tensor): cls输出, [bs, na * nc, h, w]
                bbox_pred (Tensor): reg输出, [bs, na * 4, h, w]
                coeff_pred (Tensor): mask输出, [bs, na * num_protos, h, w]
        """
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        coeff_pred = self.conv_coeff(x).tanh()
        return cls_score, bbox_pred, coeff_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """实际上该loss就是``AnchorHead.loss`` 和 ``SSDHead.loss``的组合.

        当 use_ohem为True时, 类似于 ``SSDHead.loss``,否则类似于 ``AnchorHead.loss``.
            此外,它还返回``sampling_results``.

        Args:
            cls_scores (list[Tensor]): [[bs, na * nc, h, w],] * num_level
            bbox_preds (list[Tensor]): [[bs, na * 4, h, w],] * num_level
            gt_bboxes (list[Tensor]): [[num_gts, 4], ] * bs,格式为[x1, y1, x2, y2].
            gt_labels (list[Tensor]): gt_bboxes对应的label, [[num_gts, ], ] * bs
            img_metas (list[dict]): [dict(),] * bs dict为图片的元信息.
            gt_bboxes_ignore (None | list[Tensor]): 计算损失时可以指定忽略哪些gt box.

        Returns:
            tuple:
                dict[str, Tensor]: loss的计算结果.
                List[:obj:``SamplingResult``]: [单张图像采样结果] * bs.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # [[[h * w * na, 4], ] * num_levels,] * bs,  [[[h * w * na, ], ] * num_levels, ] * bs
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
            # YOLACT中use_ohem默认为True,所以这里的cls_reg_target会基于valid_flag
            # 但cls_socre仍然是全部anchor的cls_score,因此这里会在计算cls loss时造成
            # 维度不匹配的错误,不过默认配置并不会触发该错误,但这仍然是一个待解决的问题. TODO
            # 如果要彻底解决这个潜在问题需要在loss_single_OHEM中再生成一边valid_flag
            # 及allowed_border再对pred_score与pred_reg替换.同时cls_reg_targets
            # 前四个需要重写self.get_targets内部逻辑使得数据格式为[?, ] * bs.?表示
            # 各维度数据在有效anchor上的target
            unmap_outputs=not self.use_ohem,
            # unmap_outputs=True,
            return_sampling_results=True)
        if cls_reg_targets is None:
            return None
        # [[bs, h * w * na], ] * num_level,  [[bs, h * w * na, 4],] * num_level
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, sampling_results) = cls_reg_targets

        if self.use_ohem:
            num_images = len(img_metas)
            all_cls_scores = torch.cat([
                # [bs, na * nc, h, w] -> [bs, h, w, na*nc] -> [bs, h*w*na, nc]
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, self.cls_out_channels) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1)
            all_label_weights = torch.cat(label_weights_list, -1)
            all_bbox_preds = torch.cat([
                b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
                for b in bbox_preds
            ], -2)
            all_bbox_targets = torch.cat(bbox_targets_list, -2)
            all_bbox_weights = torch.cat(bbox_weights_list, -2)

            # 将所有层级上的anchor合并到单个Tensor上
            all_anchors = []
            for i in range(num_images):
                all_anchors.append(torch.cat(anchor_list[i]))

            # 检查是否存在INF和NaN
            assert torch.isfinite(all_cls_scores).all().item(), \
                'cls score中存在INF 或 NaN!'
            assert torch.isfinite(all_bbox_preds).all().item(), \
                'box loc中存在INF 或 NaN!'

            losses_cls, losses_bbox = multi_apply(
                self.loss_single_OHEM,
                all_cls_scores,     # [bs, num_level*(h*w*na), nc]
                all_bbox_preds,     # [bs, num_level*(h*w*na), 4]
                all_anchors,        # [[num_level*(h*w*na), 4],] * bs
                all_labels,         # [bs, num_level*(h*w*na)]
                all_label_weights,  # [bs, num_level*(h*w*na)]
                all_bbox_targets,   # [bs, num_level*(h*w*na), 4]
                all_bbox_weights,   # [bs, num_level*(h*w*na), 4]
                num_total_samples=num_total_pos)
        else:
            num_total_samples = (
                num_total_pos +
                num_total_neg if self.sampling else num_total_pos)

            # 各层级上的anchor数量 [num_anchor_per_level, ] * num_level
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # [[num_level*h*w*na, 4], ] * bs -> [[bs, h*w*na, 4], ] * num_level
            concat_anchor_list = []
            for i in range(len(anchor_list)):  # range(bs)
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                               num_level_anchors)
            losses_cls, losses_bbox = multi_apply(
                self.loss_single,
                cls_scores,          # [[bs, na * nc, h, w],] * num_level
                bbox_preds,          # [[bs, na * 4, h, w],] * num_level
                all_anchor_list,     # [[bs, h * w * na, 4], ] * num_level
                labels_list,         # [[bs, h * w * na], ] * num_level
                label_weights_list,  # [[bs, h * w * na], ] * num_level
                bbox_targets_list,   # [[bs, h * w * na, 4],] * num_level
                bbox_weights_list,   # [[bs, h * w * na, 4],] * num_level
                num_total_samples=num_total_samples)
            for i in range(len(losses_cls)):
                losses_cls[i] = losses_cls[i].mean()

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox), sampling_results

    def loss_single_OHEM(self, cls_score, bbox_pred, anchors, labels,
                         label_weights, bbox_targets, bbox_weights,
                         num_total_samples):
        """"See func:``SSDHead.loss``.
        Args:
            cls_score (Tensor): [num_level*(h*w*na), nc], 所有层级上的特征点的cls输出
            bbox_pred (Tensor): [num_level*(h*w*na), 4], 所有层级上的特征点的reg输出
            anchors (Tensor): [num_level*(h*w*na), 4], 所有层级上的特征点生成的anchor
            labels (Tensor): [num_level*(h*w*na),], 所有层级上的特征点的target_cls
            label_weights (Tensor]): [num_level*(h*w*na),], 计算cls loss时该点的权重.
            bbox_targets (Tensor): [num_level*(h*w*na), 4], 所有层级上的特征点的target_reg
            bbox_weights (Tensor): [num_level*(h*w*na), 4], 计算reg loss时该点的权重.
            num_total_samples (int): batch幅图像中的正样本数
        """
        # inside_flags = anchor_inside_flags(anchors, valid_flags,
        #                                    img_meta['img_shape'][:2],
        #                                    self.train_cfg.allowed_border)
        loss_cls_all = self.loss_cls(cls_score, labels, label_weights)

        # 前景id: [0, num_classes -1], 背景id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        if num_pos_samples == 0:
            num_neg_samples = neg_inds.size(0)
        else:
            # 控制正负样本比例
            num_neg_samples = self.train_cfg.neg_pos_ratio * num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples
        if self.reg_decoded_bbox:
            # 当reg loss时IOU类 loss时,它计算predict_box与target_box之间的坐标差距
            # 否则,它计算predict_loc与target_loc之间的差距,也即修正系数的差距
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'coeff_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   coeff_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        """类似于:``AnchorHead.get_bboxes``, 但需要额外处理coeff_preds.

        Args:
            cls_scores (list[Tensor]): 所有层级的box score,
                [[bs, na * nc, h, w],] * num_level
            bbox_preds (list[Tensor]): 所有层级的box reg,
                [[bs, na * 4, h, w],] * num_level
            coeff_preds (list[Tensor]): 所有层级的Mask coefficients
                [[bs, na * num_protos, h, w],] * num_level
            img_metas (list[dict]): [dict()] * bs, batch幅图像的元信息
            cfg (mmcv.Config | None): 测试/后处理配置,如果没有,将使用 test_cfg
            rescale (bool): 是否将box缩放回原始图像尺寸下.

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: [(Tensor, Tensor, Tensor),] * bs.
                [max_per_img, 5], [max_per_img,], [max_per_img, num_protos].
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # 因为batch幅图像尺寸都一致,所以只要在一张特征图上生成一次anchor即可复用batch次.
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        det_bboxes = []
        det_labels = []
        det_coeffs = []
        for img_id in range(len(img_metas)):
            # 获取单幅图片的所有层级上的特征表示(score, reg等),
            # [[na * (nc或4或num_protos), h, w],] * num_level.
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            bbox_res = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                               coeff_pred_list, mlvl_anchors,
                                               img_shape, scale_factor, cfg,
                                               rescale)
            det_bboxes.append(bbox_res[0])
            det_labels.append(bbox_res[1])
            det_coeffs.append(bbox_res[2])
        return det_bboxes, det_labels, det_coeffs

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           coeff_preds_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """类似于:``AnchorHead._get_bboxes_single``, 但需要额外处理 coeff_preds_list,
         同时使用fast-NMS代替NMS.将单个图像的输出转换为 predict_bbox

        Args:
            cls_score_list (list[Tensor]): 所有层级的box score,
                [[na * nc, h, w],] * num_level.
            bbox_pred_list (list[Tensor]): 所有层级的box reg,
                [[na * 4, h, w],] * num_level
            coeff_preds_list (list[Tensor]): 所有层级的Mask coefficients
                [[na * num_protos, h, w],] * num_level
            mlvl_anchors (list[Tensor]): 所有层级的anchor,
                [[h * w * na, 4], ] * num_level.
            img_shape (tuple[int]): batch幅图像对齐之前的尺寸,也即pipline中Resize后的尺寸,
                [height, width, 3].
            scale_factor (ndarray): pipline中Resize操作对图像宽高的缩放系数,
                [w_scale, h_scale, w_scale, h_scale].
            cfg (mmcv.Config): 测试/后处理配置,如果没有,将使用 test_cfg.
            rescale (bool): 是否将box缩放回原始图像尺寸下.

        Returns:
            tuple[Tensor, Tensor, Tensor]: det_bboxes, det_labels, det_coeffs
                [max_per_img, 5], [max_per_img,], [max_per_img, num_protos]

        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        nms_pre = cfg.get('nms_pre', -1)
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

            if 0 < nms_pre < scores.shape[0]:
                # 获得前景类的最高分.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # 前景id: [0, num_class-1], 背景id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]  # 最大为[nms_pre, 4],下同
                bbox_pred = bbox_pred[topk_inds, :]  # [nms_pre, 4]
                # [nms_pre, nc] YOLACT的配置文件中use_sigmoid默认为False
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]  # [nms_pre, num_protos]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)
        mlvl_bboxes = torch.cat(mlvl_bboxes)  # 最大为[num_level*nms_pre, 4],下同
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)  # [num_level*nms_pre, nc]
        mlvl_coeffs = torch.cat(mlvl_coeffs)  # [num_level*nms_pre, num_protos]
        if self.use_sigmoid_cls:
            # 网络cls分支输出的激活函数为sigmoid时额外添加一个背景类,YOLACT默认为softmax.
            # 此处仅是为了格式上对齐的占位操作.
            # 前景id: [0, num_class-1], 背景id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(mlvl_bboxes, mlvl_scores,
                                                      mlvl_coeffs,
                                                      cfg.score_thr,
                                                      cfg.iou_thr, cfg.top_k,
                                                      cfg.max_per_img)
        return det_bboxes, det_labels, det_coeffs


@HEADS.register_module()
class YOLACTSegmHead(BaseModule):
    """YOLACT segmentation head used in https://arxiv.org/abs/1904.02689.

    在特征空间上使用一个仅在训练期间评估的卷积层,然后对该层输出应用语义分割损失,
    以提高性能而不会降低速度.

    Args:
        in_channels (int): 输入特征图的维度.一般为256,FPN各个输出特征图维度都是此值.
        num_classes (int): 检测类别数,不包括背景类.
        loss_segm (dict): seg loss的配置文件.
        init_cfg (dict or list[dict], optional): 该部分初始化配置文件.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 loss_segm=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='segm_conv'))):
        super(YOLACTSegmHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss_segm = build_loss(loss_segm)
        self._init_layers()
        self.fp16_enabled = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        """seg head的前向传播函数.

        Args:
            x (Tensor): 输入特征图,[bs, in_channels, h, w]

        Returns:
            Tensor: seg head预测的分割特征图, [bs, nc, h, w].
        """
        return self.segm_conv(x)

    @force_fp32(apply_to=('segm_pred', ))
    def loss(self, segm_pred, gt_masks, gt_labels):
        """Compute loss of the head.需要注意的是下面的//操作在PyTorch中是向上取整的.

        Args:
            segm_pred (Tensor): 来自Seg Head的输出特征图, [bs, nc, H//8, W//8].
            gt_masks (list[Tensor]): [[num_gt, H, W],] * bs
            gt_labels (list[Tensor]): [[num_gt,],] * bs.

        Returns:
            dict[str, Tensor]: 计算的loss字典结果.
        """
        loss_segm = []
        num_imgs, num_classes, mask_h, mask_w = segm_pred.size()
        for idx in range(num_imgs):
            cur_segm_pred = segm_pred[idx]
            cur_gt_masks = gt_masks[idx].float()
            cur_gt_labels = gt_labels[idx]
            segm_targets = self.get_targets(cur_segm_pred, cur_gt_masks,
                                            cur_gt_labels)
            if segm_targets is None:
                loss = self.loss_segm(cur_segm_pred,
                                      torch.zeros_like(cur_segm_pred),
                                      torch.zeros_like(cur_segm_pred))
            else:
                loss = self.loss_segm(
                    cur_segm_pred,
                    segm_targets,
                    avg_factor=num_imgs * mask_h * mask_w)
            loss_segm.append(loss)
        return dict(loss_segm=loss_segm)

    def get_targets(self, segm_pred, gt_masks, gt_labels):
        """计算单张图像上的seg target.简单来说就是生成一个shape和segm_pred
            一致的全为0的特征图记为segm_targets,然后将gt mask缩放至segm_pred相同尺寸,
            然后对缩放后的gt mask以0.5为阈值进行二值化.
            最后把gt mask上对应cls的值复制到segm_targets的对应cls的特征图上去,
            如果有两块mask区域重合,则取最大值.

        Args:
            segm_pred (Tensor): 预测的mask, [nc, H//8, W//8].
            gt_masks (Tensor): [num_gt, H, W].
            gt_labels (Tensor): [num_gt,].

        Returns:
            Tensor: target mask, [nc, H//8, W//8].
        """
        if gt_masks.size(0) == 0:
            return None
        num_classes, mask_h, mask_w = segm_pred.size()
        with torch.no_grad():
            downsampled_masks = F.interpolate(
                gt_masks.unsqueeze(0), (mask_h, mask_w),
                mode='bilinear',
                align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()
            segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
            for obj_idx in range(downsampled_masks.size(0)):
                segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                    segm_targets[gt_labels[obj_idx] - 1],
                    downsampled_masks[obj_idx])
            return segm_targets

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        raise NotImplementedError(
            'simple_test of YOLACTSegmHead is not implemented '
            'because this head is only evaluated during training')


@HEADS.register_module()
class YOLACTProtonet(BaseModule):
    """YOLACT mask head used in https://arxiv.org/abs/1904.02689.

    This head outputs the mask prototypes for YOLACT.

    Args:
        in_channels (int): 输入特征图的维度.
        proto_channels (tuple[int]): protonet conv的输出维度.
            需要注意的是,该参数最后一个是取决于配置文件中mask_head下面的
            num_protos值,而非32.只是配置文件中num_protos默认值为32而已.
            如果num_protos更改,那么该元组最后一个值也必须进行更改.
        proto_kernel_sizes (tuple[int]): protonet conv的卷积核大小.
        include_last_relu (Bool): 是否保留protonet的最后一个relu.
        num_protos (int): prototypes的数量.
        num_classes (int): 检测类别数.
        loss_mask_weight (float): 通过这个参数重新对mask loss加权.
        max_masks_to_train (int): 每张图像训练的最大mask数量.
        init_cfg (dict or list[dict], optional): 该层权重初始化配置.
    """

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 proto_channels=(256, 256, 256, None, 256, 32),
                 proto_kernel_sizes=(3, 3, 3, -2, 3, 1),
                 include_last_relu=True,
                 num_protos=32,
                 loss_mask_weight=1.0,
                 max_masks_to_train=100,
                 init_cfg=dict(
                     type='Xavier',
                     distribution='uniform',
                     override=dict(name='protonet'))):
        super(YOLACTProtonet, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.proto_kernel_sizes = proto_kernel_sizes
        self.include_last_relu = include_last_relu
        self.protonet = self._init_layers()

        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.fp16_enabled = False

    def _init_layers(self):
        """A helper function to take a config setting and turn it into a
        network."""
        # 不同参数组合对应不同的算子操作:
        # ( 256, 3) -> conv
        # ( 256,-2) -> deconv
        # (None,-2) -> bilinear interpolate
        in_channels = self.in_channels
        protonets = ModuleList()
        for num_channels, kernel_size in zip(self.proto_channels,
                                             self.proto_kernel_sizes):
            if kernel_size > 0:
                layer = nn.Conv2d(
                    in_channels,
                    num_channels,
                    kernel_size,
                    padding=kernel_size // 2)
            else:
                if num_channels is None:
                    layer = InterpolateModule(
                        scale_factor=-kernel_size,
                        mode='bilinear',
                        align_corners=False)
                else:
                    layer = nn.ConvTranspose2d(
                        in_channels,
                        num_channels,
                        -kernel_size,
                        padding=kernel_size // 2)
            protonets.append(layer)
            protonets.append(nn.ReLU(inplace=True))
            in_channels = num_channels if num_channels is not None \
                else in_channels
        if not self.include_last_relu:
            protonets = protonets[:-1]
        return nn.Sequential(*protonets)

    def forward_dummy(self, x):
        prototypes = self.protonet(x)
        return prototypes

    def forward(self, x, coeff_pred, bboxes, img_meta, sampling_results=None):
        """对输入特征图进行前向传播以得到prototypes, 然后使用coeff_pred与其线性组合已得到分割结果.
        最后对分割结果进行剪裁,bboxes内的保留,外部的全部为0.

        Args:
            x (Tensor): 输入特征图,如果含有FPN结构则为其中最大尺寸特征图, [bs, c, H//8, W//8]
            coeff_pred (list[Tensor]): 后续以Train为例,此时为多层级的Mask coefficients
                在Train时为predict_coeffs, [[bs, na * num_protos, h, w],] * num_level.
                在Test时为det_coeffs, [[max_per_img, num_protos],] * bs.
            bboxes (list[Tensor]): 后续以Train为例.
                在Train时为gt box,[[num_gts, 4],] * bs.
                在Test时为det_box.[[max_per_img, * 4],] * bs. Resize后的图像尺寸下的坐标
            img_meta (list[dict]): [dict(),] * bs,batch幅图像的元信息.
            sampling_results (List[:obj:``SamplingResult``]): batch幅图像的采样结果.

        Returns:
            list[Tensor]: Predicted mask, [[?, H//4, W//4],] * bs.
                ? 在Train/val时为num_pos,在Test时为max_per_img
        """
        # [bs, c, H//8, W//8] -> [bs, 32, H//4, W//4] -> [bs, H//4, W//4, num_protos]
        prototypes = self.protonet(x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = x.size(0)

        # 不使用self.training的原因是在val时会出现维度不匹配的错误.
        # 值得注意的是,这个写法是非常具有技巧性的的.
        # Fix https://github.com/open-mmlab/mmdetection/issues/5978
        is_train_or_val_workflow = (coeff_pred[0].dim() == 4)

        # Train or val workflow  指的是default_runtime.py中的workflow
        if is_train_or_val_workflow:
            coeff_pred_list = []
            for coeff_pred_per_level in coeff_pred:
                # [bs, h, w, na * num_protos] -> [bs, h*w*na, num_protos]
                coeff_pred_per_level = \
                    coeff_pred_per_level.permute(
                        0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
                coeff_pred_list.append(coeff_pred_per_level)
            # [bs, num_level*(h*w*na), num_protos]
            coeff_pred = torch.cat(coeff_pred_list, dim=1)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            cur_coeff_pred = coeff_pred[idx]
            cur_bboxes = bboxes[idx]
            cur_img_meta = img_meta[idx]

            # Testing state
            if not is_train_or_val_workflow:
                bboxes_for_cropping = cur_bboxes
            else:
                cur_sampling_results = sampling_results[idx]
                # 这里可以使用cur_sampling_results.pos_gt_bboxes TODO 待替换
                pos_assigned_gt_inds = \
                    cur_sampling_results.pos_assigned_gt_inds
                bboxes_for_cropping = cur_bboxes[pos_assigned_gt_inds].clone()
                pos_inds = cur_sampling_results.pos_inds
                cur_coeff_pred = cur_coeff_pred[pos_inds]

            # ? 在Train/val时为num_pos,在Test时为max_per_img
            # 将 prototypes 和 coeff_pred线性组合得到[H//4, W//4, ?]
            # [H//4, W//4, num_protos] @ [num_protos, ?]
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
        """Compute loss of the head.
        注意num_pos取决于prior与gt的分布,但其与num_gt的大小是不确定的,可大可小可等于.
        小于是由于部分gt与prior的IOU小于pos_iou_thr所致.

        Args:
            mask_pred (list[Tensor]): [[num_pos, H//4, W//4], ] * bs
            gt_masks (list[Tensor]): [[num_gt, H, W], ] * bs
            gt_bboxes (list[Tensor]): [[num_gt, 4], ] * bs, [x1, y1, x2, y2]格式.
            img_meta (list[dict]): [dict(),] * bs,batch幅图像的元信息.
            sampling_results (List[:obj:``SamplingResult``]): batch幅图像的采样结果.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
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
            # 如果我们在所有的正样本mask上进行反向传播需要较多的显存.因此如果正样本数量太多.
            # 那么就随机选取其中一部分的正样本mask来进行反向传播
            if num_pos > self.max_masks_to_train:
                perm = torch.randperm(num_pos)
                select = perm[:self.max_masks_to_train]
                cur_mask_pred = cur_mask_pred[select]
                pos_assigned_gt_inds = pos_assigned_gt_inds[select]
                num_pos = self.max_masks_to_train
            total_pos += num_pos

            gt_bboxes_for_reweight = cur_gt_bboxes[pos_assigned_gt_inds]

            # 获取正样本所对应的target_mask
            mask_targets = self.get_targets(cur_mask_pred, cur_gt_masks,
                                            pos_assigned_gt_inds)
            if num_pos == 0:
                loss = cur_mask_pred.sum() * 0.
            elif mask_targets is None:
                loss = F.binary_cross_entropy(cur_mask_pred,
                                              torch.zeros_like(cur_mask_pred),
                                              torch.zeros_like(cur_mask_pred))
            else:
                # cur_mask_pred是经过sigmoid输出的,理论上是∈(0, 1)的.
                # 但sigmoid中有指数操作,可能会出现INF,所以这还是需要限制一下的.
                cur_mask_pred = torch.clamp(cur_mask_pred, 0, 1)
                loss = F.binary_cross_entropy(
                    cur_mask_pred, mask_targets,
                    reduction='none') * self.loss_mask_weight

                h, w = cur_img_meta['img_shape'][:2]
                gt_bboxes_width = (gt_bboxes_for_reweight[:, 2] -
                                   gt_bboxes_for_reweight[:, 0]) / w
                gt_bboxes_height = (gt_bboxes_for_reweight[:, 3] -
                                    gt_bboxes_for_reweight[:, 1]) / h
                # 这里对不同大小的gt的loss进行加权
                loss = loss.mean(dim=(1,
                                      2)) / gt_bboxes_width / gt_bboxes_height
                loss = torch.sum(loss)
            loss_mask.append(loss)

        if total_pos == 0:
            total_pos += 1  # 避免除数为0导致NaN
        loss_mask = [x / total_pos for x in loss_mask]

        return dict(loss_mask=loss_mask)

    def get_targets(self, mask_pred, gt_masks, pos_assigned_gt_inds):
        """计算每个图像的mask target.将gt mask长宽都缩放至mask_pred尺寸,然后大于0.5
        的区域都置为1,否则为0.最后将正样本所对应的gt索引再对应的gt mask作为seg target返回.

        Args:
            mask_pred (Tensor): [num_pos, H//4, W//4].
            gt_masks (Tensor): [num_gt, H, W].
            pos_assigned_gt_inds (Tensor): [num_pos,].正样本所对应的gt的索引
        Returns:
            mask_targets (Tensor): seg target,[num_pos, H//4, W//4].
        """
        if gt_masks.size(0) == 0:
            return None
        mask_h, mask_w = mask_pred.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        mask_targets = gt_masks[pos_assigned_gt_inds]
        return mask_targets

    def get_seg_masks(self, mask_pred, label_pred, img_meta, rescale):
        """缩放回目标尺寸, 二值化, 格式化mask.

        Args:
            mask_pred (Tensor): [max_per_img, H//4, W//4].
            label_pred (Tensor): [max_per_img, ].
            img_meta (dict): [dict(),] * bs. dict()为图像元信息.
            rescale (bool): 是否将mask缩放至原始图像尺寸下.
        Returns:
            list[ndarray]: 按分割类别分组的predict mask. mask_h或mask_w取决于rescale.
                [[[mask_h, mask_w], ] * num_seg_per_cls] * nc.
        """
        ori_shape = img_meta['ori_shape']
        scale_factor = img_meta['scale_factor']
        # 如果需要缩放回Resize前尺寸,那么直接跳过[H,W],直接从[H//4,W//4] resize到[ori_h,ori_w]
        # 否则就resize到网络输入尺寸[H, W]
        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor[1]).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor[0]).astype(np.int32)

        cls_segms = [[] for _ in range(self.num_classes)]
        if mask_pred.size(0) == 0:
            return cls_segms

        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > 0.5
        mask_pred = mask_pred.cpu().numpy().astype(np.uint8)

        for m, l in zip(mask_pred, label_pred):
            cls_segms[l].append(m)
        return cls_segms

    def crop(self, masks, boxes, padding=1):
        """box坐标本身是归一化的,先将其坐标放缩回mask尺寸下,
            再将box中的所有mask区域归零.

        Args:
            masks (Tensor): shape [h, batch_w//4, batch_h//4].
            boxes (Tensor): [n, 4].相对坐标,∈[0, 1]

        Return:
            Tensor: 裁剪后的mask.
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

    def sanitize_coordinates(self, x1, x2, img_size, padding=0, cast=True):
        """清理及限制输入坐标,使 x1 < x2, x1 != x2, x1 >= 0 和 x2 <= image_size.
        还将相对坐标转换为绝对坐标并将结果转换为长Long型Tensor.

        Warning: 以下操作为in-place,因此如有必要.请复制.

        Args:
            _x1 (Tensor): shape (N, ).
            _x2 (Tensor): shape (N, ).
            img_size (int): 输入图像的尺寸.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): 如果为False, 返回值将不会转为Long型Tensor.

        Returns:
            tuple:
                x1 (Tensor): Sanitized _x1.
                x2 (Tensor): Sanitized _x2.
        """
        x1 = x1 * img_size
        x2 = x2 * img_size
        if cast:
            x1 = x1.long()
            x2 = x2.long()
        x1 = torch.min(x1, x2)
        x2 = torch.max(x1, x2)
        x1 = torch.clamp(x1 - padding, min=0)
        x2 = torch.clamp(x2 + padding, max=img_size)
        return x1, x2

    def simple_test(self,
                    feats,
                    det_bboxes,
                    det_labels,
                    det_coeffs,
                    img_metas,
                    rescale=False):
        """非TTA模式.

        Args:
            feats (tuple[torch.Tensor]): 所有层级的特征图.
                [[bs, c, h, w], ] * num_level
            det_bboxes (list[Tensor]): [[max_per_img, 5], ] * bs
            det_labels (list[Tensor]): [[max_per_img, ], ] * bs.
            det_coeffs (list[Tensor]): [[max_per_img, num_protos], ] * bs.
            img_metas (list[dict]): [dict(),] * bs. dict()为图像元信息.
            rescale (bool, optional): 是否将mask缩放回原始图像尺寸空间中.

        Returns:
            list[list]: 整个batch图像的分割结果,img_h或img_w取决于rescale.
                rescale为True时, mask_h为图像Resize前高度, mask_w同理
                rescale为False时, mask_h为图像Resize后高度, mask_w同理
                [[[[mask_h, mask_w],] * num_seg_per_cls] * nc] * bs.
        """
        num_imgs = len(img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # 如果box已经缩放回Resize前尺寸,则需将其重新放缩回Resize后尺寸以获得seg.
            if rescale and not isinstance(scale_factors[0], float):
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_preds = self.forward(feats[0], det_coeffs, _bboxes, img_metas)
            # 分别对每个图像应用mask后处理
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append([[] for _ in range(self.num_classes)])
                else:
                    segm_result = self.get_seg_masks(mask_preds[i],
                                                     det_labels[i],
                                                     img_metas[i], rescale)
                    segm_results.append(segm_result)
        return segm_results


class InterpolateModule(BaseModule):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, init_cfg=None, **kwargs):
        super().__init__(init_cfg)

        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        """Forward features from the upstream network."""
        return F.interpolate(x, *self.args, **self.kwargs)
