# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from ..layers import fast_nms
from ..utils import images_to_levels, multi_apply, select_single_mlvl
from ..utils.misc import empty_instances
from .anchor_head import AnchorHead
from .base_mask_head import BaseMaskHead


@MODELS.register_module()
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
        anchor_generator (:obj:`ConfigDict` or dict): anchor生成策略的配置文件
        loss_cls (:obj:`ConfigDict` or dict): cls loss的配置文件.
        loss_bbox (:obj:`ConfigDict` or dict): reg loss的配置文件.
        num_head_convs (int): box 和 cls 分支共享的卷积层数.
        num_protos (int): mask coefficients的数量.
        use_ohem (bool): 如果为true, 则使用``loss_single_OHEM``计算cls loss
            否则使用``loss_single``.
        conv_cfg (:obj:`ConfigDict` or dict, optional): 构建和配置conv层的字典.
        norm_cfg (:obj:`ConfigDict` or dict, optional): 构建和配置norm层的字典.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): 模型权重初始化的配置字典.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 anchor_generator: ConfigType = dict(
                     type='AnchorGenerator',
                     octave_base_scale=3,
                     scales_per_octave=1,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     reduction='none',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
                 num_head_convs: int = 1,
                 num_protos: int = 32,
                 use_ohem: bool = True,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = dict(
                     type='Xavier',
                     distribution='uniform',
                     bias=0,
                     layer='Conv2d'),
                 **kwargs) -> None:
        self.num_head_convs = num_head_convs
        self.num_protos = num_protos
        self.use_ohem = use_ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self) -> None:
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

    def forward_single(self, x: Tensor) -> tuple:
        """单层级上的前向传播.

        Args:
            x (Tensor): 单层级特征图.

        Returns:
            tuple:

            - cls_score (Tensor): cls输出, [bs, na * nc, h, w]
            - bbox_pred (Tensor): reg输出, [bs, na * 4, h, w]
            - coeff_pred (Tensor): mask输出, [bs, na * num_protos, h, w]
        """
        for head_conv in self.head_convs:
            x = head_conv(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        coeff_pred = self.conv_coeff(x).tanh()
        return cls_score, bbox_pred, coeff_pred

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            coeff_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """基于box head输出的特征图计算loss

        当 use_ohem为True时, 类似于 ``SSDHead.loss``,否则类似于 ``AnchorHead.loss``.
            此外,它还返回``sampling_results``.

        Args:
            cls_scores (list[Tensor]): [[bs, na * nc, h, w],] * nl
            bbox_preds (list[Tensor]): [[bs, na * 4, h, w],] * nl
            coeff_preds (list[Tensor]): [[bs, na * num_proto, h, w],] * nl
                level with shape (N, num_anchors * num_protos, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): bs幅图像的有效标注信息.
            batch_img_metas (list[dict]): [dict(),] * bs bs幅图像的元信息.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                bs幅图像的忽略标注信息.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        # [[[h * w * na, 4], ] * nl,] * bs,  [[[h * w * na, ], ] * nl, ] * bs
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            # YOLACT中use_ohem默认为True,所以这里的cls_reg_target会基于valid_flag
            # 但cls_socre仍然是全部anchor的cls_score,因此这里会在计算cls loss时造成
            # 维度不匹配的错误,不过默认配置并不会触发该错误,但这仍然是一个待解决的问题. TODO
            # 如果要彻底解决这个潜在问题需要在loss_single_OHEM中再生成一边valid_flag
            # 及allowed_border再对pred_score与pred_reg替换.同时cls_reg_targets
            # 前四个需要重写self.get_targets内部逻辑使得数据格式为[?, ] * bs.?表示
            # 各维度数据在有效anchor上的target
            unmap_outputs=not self.use_ohem,
            return_sampling_results=True)
        # [[bs, h * w * na], ] * nl,  [[bs, h * w * na, 4],] * nl
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor, sampling_results) = cls_reg_targets

        if self.use_ohem:
            num_images = len(batch_img_metas)
            all_cls_scores = torch.cat([
                # [bs, na * nc, h, w] -> [bs, h, w, na*nc] -> [bs, h*w*na, nc]
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
                self.OHEMloss_by_feat_single,
                all_cls_scores,     # [bs, nl*(h*w*na), nc]
                all_bbox_preds,     # [bs, nl*(h*w*na), 4]
                all_anchors,        # [[nl*(h*w*na), 4],] * bs
                all_labels,         # [bs, nl*(h*w*na)]
                all_label_weights,  # [bs, nl*(h*w*na)]
                all_bbox_targets,   # [bs, nl*(h*w*na), 4]
                all_bbox_weights,   # [bs, nl*(h*w*na), 4]
                avg_factor=avg_factor)
        else:
            # 各层级上的anchor数量 [num_anchor_per_level, ] * num_level
            num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
            # [[nl*h*w*na, 4], ] * bs -> [[bs, h*w*na, 4], ] * nl
            concat_anchor_list = []
            for i in range(len(anchor_list)):
                concat_anchor_list.append(torch.cat(anchor_list[i]))
            all_anchor_list = images_to_levels(concat_anchor_list,
                                               num_level_anchors)
            losses_cls, losses_bbox = multi_apply(
                self.loss_by_feat_single,
                cls_scores,             # [[bs, na * nc, h, w],] * nl
                bbox_preds,             # [[bs, na * 4, h, w],] * nl
                all_anchor_list,        # [[bs, h * w * na, 4], ] * nl
                labels_list,            # [[bs, h * w * na], ] * nl
                label_weights_list,     # [[bs, h * w * na], ] * nl
                bbox_targets_list,      # [[bs, h * w * na, 4],] * nl
                bbox_weights_list,      # [[bs, h * w * na, 4],] * nl
                avg_factor=avg_factor)
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        # update `_raw_positive_infos`, which will be used when calling
        # `get_positive_infos`.
        self._raw_positive_infos.update(coeff_preds=coeff_preds)
        return losses

    def OHEMloss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                                anchors: Tensor, labels: Tensor,
                                label_weights: Tensor, bbox_targets: Tensor,
                                bbox_weights: Tensor,
                                avg_factor: int) -> tuple:
        """Compute loss of a single image. Similar to
        func:``SSDHead.loss_by_feat_single``

        Args:
            cls_score (Tensor): [nl*(h*w*na), nc], cls score
            bbox_pred (Tensor): [nl*(h*w*na), 4], box reg
            anchors (Tensor): [nl*(h*w*na), 4], anchor
            labels (Tensor): [nl*(h*w*na),], target cls
            label_weights (Tensor): [nl*(h*w*na),], anchor计算cls loss时的权重.
            bbox_targets (Tensor): [nl*(h*w*na), 4], target reg
            bbox_weights (Tensor): anchor计算reg loss时的权重.
            avg_factor (int): 平均因子.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of cls loss and bbox loss of one
            feature map.
        """

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
            num_neg_samples = self.train_cfg['neg_pos_ratio'] * \
                              num_pos_samples
            if num_neg_samples > neg_inds.size(0):
                num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor
        if self.reg_decoded_bbox:
            # 当reg loss时IOU类 loss时,它计算predict_box与target_box之间的坐标差距
            # 否则,它计算predict_loc与target_loc之间的差距,也即修正系数的差距
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls[None], loss_bbox

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive Information of each image,
            usually including positive bboxes, positive labels, positive
            priors, positive coeffs, etc.
        """
        assert len(self._raw_positive_infos) > 0
        sampling_results = self._raw_positive_infos['sampling_results']
        num_imgs = len(sampling_results)

        coeff_pred_list = []
        for coeff_pred_per_level in self._raw_positive_infos['coeff_preds']:
            # [bs, h, w, na * num_protos] -> [bs, h*w*na, num_protos]
            coeff_pred_per_level = \
                coeff_pred_per_level.permute(
                    0, 2, 3, 1).reshape(num_imgs, -1, self.num_protos)
            coeff_pred_list.append(coeff_pred_per_level)
        # [bs, num_level*(h*w*na), num_protos]
        coeff_preds = torch.cat(coeff_pred_list, dim=1)

        pos_info_list = []
        for idx, sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            coeff_preds_single = coeff_preds[idx]
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            pos_info.coeffs = coeff_preds_single[sampling_result.pos_inds]
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info_list.append(pos_info)
        return pos_info_list

    def predict_by_feat(self,
                        cls_scores,
                        bbox_preds,
                        coeff_preds,
                        batch_img_metas,
                        cfg=None,
                        rescale=True,
                        **kwargs):
        """Similar to func:``AnchorHead.get_bboxes``, but additionally
        processes coeff_preds.

        Args:
            cls_scores (list[Tensor]): 所有层级的box score,
                [[bs, na * nc, h, w],] * nl
            bbox_preds (list[Tensor]): 所有层级的box reg,
                [[bs, na * 4, h, w],] * nl
            coeff_preds (list[Tensor]): 所有层级的Mask coefficients
                [[bs, na * num_protos, h, w],] * nl
            batch_img_metas (list[dict]): [dict()] * bs, batch幅图像的元信息
            cfg (:obj:`Config` | None): 测试/后处理配置,如果没有,将使用 test_cfg
            rescale (bool): 是否将box缩放回原始图像尺寸下.

        Returns:
            list[:obj:`InstanceData`]: batch幅图像经过后处理的检测结果. 每个元素包含以下值.
                - scores (Tensor): cls score, (max_per_img, )
                - labels (Tensor): cls label, (max_per_img, )
                - bboxes (Tensor): box (max_per_img, 4), xyxy格式.
                - coeffs (Tensor): box对应的 mask coefficients,
                  (max_per_img, num_protos).
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # 因为batch幅图像尺寸都一致,所以只要在一张特征图上生成一次anchor即可复用batch次.
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            # 获取单幅图片的所有层级上的特征表示(score, reg等),
            # [[na * (nc或4或num_protos), h, w],] * nl.
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            coeff_pred_list = select_single_mlvl(coeff_preds, img_id)
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                coeff_preds_list=coeff_pred_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                coeff_preds_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = True) -> InstanceData:
        """类似于:``AnchorHead._predict_by_feat_single``, 但需要额外处理 coeff_preds_list,
         同时使用fast-NMS代替NMS.将单个图像的输出转换为 predict_bbox
        of traditional NMS.

        Args:
            cls_score_list (list[Tensor]): 所有层级的box score,
                [[na * nc, h, w],] * nl.
            bbox_pred_list (list[Tensor]): 所有层级的box reg,
                [[na * 4, h, w],] * nl
            coeff_preds_list (list[Tensor]): 所有层级的Mask coefficients
                [[na * num_protos, h, w],] * nl
            mlvl_priors (list[Tensor]): 所有层级的prior, [[h * w * na, 4], ] * nl.
            img_meta (dict): 图像元信息.
            cfg (mmengine.Config): 测试/后处理配置,如果没有,将使用 test_cfg.
            rescale (bool): 是否将box缩放回原始图像尺寸下.

        Returns:
            :obj:`InstanceData`: batch幅图像经过后处理的检测结果. 每个元素包含以下值.

                - scores (Tensor): cls score, (max_per_img, )
                - labels (Tensor): cls label, (max_per_img, )
                - bboxes (Tensor): box (max_per_img, 4), xyxy格式.
                - coeffs (Tensor): box对应的 mask coefficients,
                  (max_per_img, num_protos).
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_coeffs = []
        for cls_score, bbox_pred, coeff_pred, priors in \
                zip(cls_score_list, bbox_pred_list,
                    coeff_preds_list, mlvl_priors):
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
                priors = priors[topk_inds, :]  # 最大为[nms_pre, 4],下同
                bbox_pred = bbox_pred[topk_inds, :]  # [nms_pre, 4]
                # [nms_pre, nc] YOLACT的配置文件中use_sigmoid默认为False
                scores = scores[topk_inds, :]
                coeff_pred = coeff_pred[topk_inds, :]  # [nms_pre, num_protos]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_coeffs.append(coeff_pred)
        # 最大为[nl*nms_pre, 4],下同
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = torch.cat(mlvl_valid_priors)
        multi_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=img_shape)
        # [nl*nms_pre, nc/num_proto]
        multi_scores = torch.cat(mlvl_scores)
        multi_coeffs = torch.cat(mlvl_coeffs)

        return self._bbox_post_process(
            multi_bboxes=multi_bboxes,
            multi_scores=multi_scores,
            multi_coeffs=multi_coeffs,
            cfg=cfg,
            rescale=rescale,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           multi_bboxes: Tensor,
                           multi_scores: Tensor,
                           multi_coeffs: Tensor,
                           cfg: ConfigType,
                           rescale: bool = False,
                           img_meta: Optional[dict] = None,
                           **kwargs) -> InstanceData:
        """box的后处理方法.
        这些box将重新缩放到原始图像尺寸并执行 nms 操作。通常 `with_nms` 为 False 用于TTA.

        Args:
            multi_bboxes (Tensor): [nl*nms_pre, 4]
            multi_scores (Tensor): [nl*nms_pre, nc]
            multi_coeffs (Tensor): [nl*nms_pre, num_proto]
            cfg (ConfigDict): 测试/后处理配置,如果没有,将使用 test_cfg.
            rescale (bool): 是否将box缩放回原始图像尺寸下.
            img_meta (dict, optional): 图像元信息.

        Returns:
            :obj:`InstanceData`: batch幅图像经过后处理的检测结果. 每个元素包含以下值.

                - scores (Tensor): cls score, (max_per_img, )
                - labels (Tensor): cls label, (max_per_img, )
                - bboxes (Tensor): box (max_per_img, 4), xyxy格式.
                - coeffs (Tensor): box对应的 mask coefficients,
                  (max_per_img, num_protos).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            multi_bboxes /= multi_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
            # mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        if self.use_sigmoid_cls:
            # 网络cls分支输出的激活函数为sigmoid时额外添加一个背景类,YOLACT默认为softmax.
            # 此处仅是为了格式上对齐的占位操作.
            # 前景id: [0, num_class-1], 背景id: num_class
            padding = multi_scores.new_zeros(multi_scores.shape[0], 1)
            multi_scores = torch.cat([multi_scores, padding], dim=1)
        det_bboxes, det_labels, det_coeffs = fast_nms(
            multi_bboxes, multi_scores, multi_coeffs, cfg.score_thr,
            cfg.iou_thr, cfg.top_k, cfg.max_per_img)
        results = InstanceData()
        results.bboxes = det_bboxes[:, :4]
        results.scores = det_bboxes[:, -1]
        results.labels = det_labels
        results.coeffs = det_coeffs
        return results


@MODELS.register_module()
class YOLACTProtonet(BaseMaskHead):
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

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        proto_channels: tuple = (256, 256, 256, None, 256, 32),
        proto_kernel_sizes: tuple = (3, 3, 3, -2, 3, 1),
        include_last_relu: bool = True,
        num_protos: int = 32,
        loss_mask_weight: float = 1.0,
        max_masks_to_train: int = 100,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        with_seg_branch: bool = True,
        loss_segm: ConfigType = dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        init_cfg=dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='protonet'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.proto_channels = proto_channels
        self.proto_kernel_sizes = proto_kernel_sizes
        self.include_last_relu = include_last_relu

        # Segmentation branch
        self.with_seg_branch = with_seg_branch
        self.segm_branch = SegmentationModule(
            num_classes=num_classes, in_channels=in_channels) \
            if with_seg_branch else None
        self.loss_segm = MODELS.build(loss_segm) if with_seg_branch else None

        self.loss_mask_weight = loss_mask_weight
        self.num_protos = num_protos
        self.num_classes = num_classes
        self.max_masks_to_train = max_masks_to_train
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
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
        self.protonet = nn.Sequential(*protonets)

    def forward(self, x: tuple, positive_infos: InstanceList) -> tuple:
        """对输入特征图进行前向传播以得到prototypes, 然后使用coeff_pred与其线性组合已得到分割结果.
        最后对分割结果进行剪裁,bboxes内的保留,外部的全部为0.

        Args:
            x (Tuple[Tensor]): 所有层级上的特征图, [[bs, c, batch_h//8, ~], ] * nl
            positive_infos (List[:obj:``InstanceData``]): Positive information
                that calculate from detect head.

        Returns:
            tuple: Predicted instance segmentation masks and
            semantic segmentation map.
        """
        # YOLACT 使用最大的特征图来得到seg mask
        single_x = x[0]

        # YOLACT的seg分支, 如果处于非训练阶段或者seg分支为None时将跳过seg分支
        if self.segm_branch is not None and self.training:
            segm_preds = self.segm_branch(single_x)
        else:
            segm_preds = None
        # YOLACT mask head
        # [bs, c, batch_h//8, ~] -> [bs, 32, batch_h//4, ~] -> [bs, batch_h//4, ~, num_proto]
        prototypes = self.protonet(single_x)
        prototypes = prototypes.permute(0, 2, 3, 1).contiguous()

        num_imgs = single_x.size(0)

        mask_pred_list = []
        for idx in range(num_imgs):
            cur_prototypes = prototypes[idx]
            pos_coeffs = positive_infos[idx].coeffs

            # ? 在Train/val时为num_pos,在Test时为max_per_img
            # 将 prototypes 和 coeff_pred线性组合得到[H//4, W//4, ?]
            # [H//4, W//4, num_protos] @ [num_protos, ?]
            mask_preds = cur_prototypes @ pos_coeffs.t()
            mask_preds = torch.sigmoid(mask_preds)
            mask_pred_list.append(mask_preds)
        return mask_pred_list, segm_preds

    def loss_by_feat(self, mask_preds: List[Tensor], segm_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict], positive_infos: InstanceList,
                     **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the mask head.

        Args:
            mask_preds (list[Tensor]): [[nc, batch_h//4, batch_w//4], ] * bs
            segm_preds (Tensor):[bs, nc, batch_h//4, batch_w//4]
            batch_gt_instances (list[:obj:`InstanceData`]): bs幅图像有效的标注信息.
                包含 ``bboxes``, ``masks``, ``labels`` 属性.
            batch_img_metas (list[dict]): batch幅图像的元信息.
            positive_infos (List[:obj:``InstanceData``]): Information of
                positive samples of each image that are assigned in detection
                head.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert positive_infos is not None, \
            'positive_infos should not be None in `YOLACTProtonet`'
        losses = dict()

        # crop
        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                positive_infos)

        loss_mask = []
        loss_segm = []
        num_imgs, _, mask_h, mask_w = segm_preds.size()
        assert num_imgs == len(croped_mask_pred)
        segm_avg_factor = num_imgs * mask_h * mask_w
        total_pos = 0

        if self.segm_branch is not None:
            assert segm_preds is not None

        for idx in range(num_imgs):
            img_meta = batch_img_metas[idx]

            (mask_preds, pos_mask_targets, segm_targets, num_pos,
             gt_bboxes_for_reweight) = self._get_targets_single(
                 croped_mask_pred[idx], segm_preds[idx],
                 batch_gt_instances[idx], positive_infos[idx])

            # segmentation loss
            if self.with_seg_branch:
                if segm_targets is None:
                    loss = segm_preds[idx].sum() * 0.
                else:
                    loss = self.loss_segm(
                        segm_preds[idx],
                        segm_targets,
                        avg_factor=segm_avg_factor)
                loss_segm.append(loss)
            # mask loss
            total_pos += num_pos
            if num_pos == 0 or pos_mask_targets is None:
                loss = mask_preds.sum() * 0.
            else:
                # cur_mask_pred是经过sigmoid输出的,理论上是∈(0, 1)的.
                # 但sigmoid中有指数操作,可能会出现INF,所以这还是需要限制一下的.
                mask_preds = torch.clamp(mask_preds, 0, 1)
                loss = F.binary_cross_entropy(
                    mask_preds, pos_mask_targets,
                    reduction='none') * self.loss_mask_weight

                h, w = img_meta['img_shape'][:2]
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

        losses.update(loss_mask=loss_mask)
        if self.with_seg_branch:
            losses.update(loss_segm=loss_segm)

        return losses

    def _get_targets_single(self, mask_preds: Tensor, segm_pred: Tensor,
                            gt_instances: InstanceData,
                            positive_info: InstanceData):
        """计算单张图像上的seg target.简单来说就是生成一个shape和segm_pred
            一致的全为0的特征图记为segm_targets,然后将gt mask缩放至segm_pred相同尺寸,
            然后对缩放后的gt mask以0.5为阈值进行二值化.
            最后把gt mask上对应cls的值复制到segm_targets的对应cls的特征图上去,
            如果有两块mask区域重合,则取最大值.

        Args:
            mask_preds (Tensor): [nc, batch_h//4, batch_w//4].
            segm_pred (Tensor): Predicted semantic segmentation map
                with shape (nc, batch_h//4, batch_w//4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes``, ``labels``,
                and ``masks`` attributes.
            positive_info (:obj:`InstanceData`): Information of positive
                samples that are assigned in detection head. It usually
                contains following keys.

                    - pos_assigned_gt_inds (Tensor): Assigner GT indexes of
                      positive proposals, has shape (num_pos, )
                    - pos_inds (Tensor): Positive index of image, has
                      shape (num_pos, ).
                    - coeffs (Tensor): Positive mask coefficients
                      with shape (num_pos, num_protos).
                    - bboxes (Tensor): Positive bboxes with shape
                      (num_pos, 4)

        Returns:
            tuple: Usually returns a tuple containing learning targets.

            - mask_preds (Tensor): Positive predicted mask with shape
              (num_pos, mask_h, mask_w).
            - pos_mask_targets (Tensor): Positive mask targets with shape
              (num_pos, mask_h, mask_w).
            - segm_targets (Tensor): Semantic segmentation targets with shape
              (num_classes, segm_h, segm_w).
            - num_pos (int): Positive numbers.
            - gt_bboxes_for_reweight (Tensor): GT bboxes that match to the
              positive priors has shape (num_pos, 4).
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        device = gt_bboxes.device
        gt_masks = gt_instances.masks.to_tensor(
            dtype=torch.bool, device=device).float()
        if gt_masks.size(0) == 0:
            return mask_preds, None, None, 0, None

        # process with semantic segmentation targets
        if segm_pred is not None:
            num_classes, segm_h, segm_w = segm_pred.size()
            with torch.no_grad():
                downsampled_masks = F.interpolate(
                    gt_masks.unsqueeze(0), (segm_h, segm_w),
                    mode='bilinear',
                    align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segm_targets = torch.zeros_like(segm_pred, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segm_targets[gt_labels[obj_idx] - 1] = torch.max(
                        segm_targets[gt_labels[obj_idx] - 1],
                        downsampled_masks[obj_idx])
        else:
            segm_targets = None
        # process with mask targets
        pos_assigned_gt_inds = positive_info.pos_assigned_gt_inds
        num_pos = pos_assigned_gt_inds.size(0)
        # 如果我们在所有的正样本mask上进行反向传播需要较多的显存.因此如果正样本数量太多.
        # 那么就随机选取其中一部分的正样本mask来进行反向传播
        if num_pos > self.max_masks_to_train:
            perm = torch.randperm(num_pos)
            select = perm[:self.max_masks_to_train]
            mask_preds = mask_preds[select]
            pos_assigned_gt_inds = pos_assigned_gt_inds[select]
            num_pos = self.max_masks_to_train

        gt_bboxes_for_reweight = gt_bboxes[pos_assigned_gt_inds]

        mask_h, mask_w = mask_preds.shape[-2:]
        gt_masks = F.interpolate(
            gt_masks.unsqueeze(0), (mask_h, mask_w),
            mode='bilinear',
            align_corners=False).squeeze(0)
        gt_masks = gt_masks.gt(0.5).float()
        pos_mask_targets = gt_masks[pos_assigned_gt_inds]

        return (mask_preds, pos_mask_targets, segm_targets, num_pos,
                gt_bboxes_for_reweight)

    def crop_mask_preds(self, mask_preds: List[Tensor],
                        batch_img_metas: List[dict],
                        positive_infos: InstanceList) -> list:
        """Crop predicted masks by zeroing out everything not in the predicted
        bbox.

        Args:
            mask_preds (list[Tensor]): Predicted prototypes with shape
                [[nc, batch_h//4, batch_h//4], ] * bs.
            batch_img_metas (list[dict]): Meta information of multiple images.
            positive_infos (List[:obj:``InstanceData``]): Positive
                information that calculate from detect head.

        Returns:
            list: 裁剪后的mask.
        """
        croped_mask_preds = []
        for img_meta, mask_preds, cur_info in zip(batch_img_metas, mask_preds,
                                                  positive_infos):
            bboxes_for_cropping = copy.deepcopy(cur_info.bboxes)
            h, w = img_meta['img_shape'][:2]
            bboxes_for_cropping[:, 0::2] /= w
            bboxes_for_cropping[:, 1::2] /= h
            mask_preds = self.crop_single(mask_preds, bboxes_for_cropping)
            mask_preds = mask_preds.permute(2, 0, 1).contiguous()
            croped_mask_preds.append(mask_preds)
        return croped_mask_preds

    def crop_single(self,
                    masks: Tensor,
                    boxes: Tensor,
                    padding: int = 1) -> Tensor:
        """box坐标本身是归一化的,先将其坐标放缩回mask尺寸下,再将不在box中的所有mask区域归零.

        Args:
            masks (Tensor): [batch_w//4, ~, N].
            boxes (Tensor): [N, 4], 相对坐标,∈[0, 1]
            padding (int): Image padding size.

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

    def sanitize_coordinates(self,
                             x1: Tensor,
                             x2: Tensor,
                             img_size: int,
                             padding: int = 0,
                             cast: bool = True) -> tuple:
        """清理及限制输入坐标,使 x1 < x2, x1 != x2, x1 >= 0 和 x2 <= image_size.
        还将相对坐标转换为绝对坐标并将结果转换为长Long型Tensor.

        Warning: 以下操作为in-place,因此如有必要.请复制再操作.

        Args:
            x1 (Tensor): shape (N, ).
            x2 (Tensor): shape (N, ).
            img_size (int): 输入图像的尺寸.
            padding (int): x1 >= padding, x2 <= image_size-padding.
            cast (bool): 如果为False, 返回值将不会转为Long型Tensor.

        Returns:
            tuple:

            - x1 (Tensor): Sanitized _x1.
            - x2 (Tensor): Sanitized _x2.
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

    def predict_by_feat(self,
                        mask_preds: List[Tensor],
                        segm_preds: Tensor,
                        results_list: InstanceList,
                        batch_img_metas: List[dict],
                        rescale: bool = True,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (list[Tensor]): [[nc, batch_h//4, batch_w//4], ] * bs
            results_list (List[:obj:``InstanceData``]): BBoxHead results.
            batch_img_metas (list[dict]): Meta information of all images.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Processed results of multiple
            images.Each :obj:`InstanceData` usually contains
            following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        croped_mask_pred = self.crop_mask_preds(mask_preds, batch_img_metas,
                                                results_list)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            mask_preds = croped_mask_pred[img_id]
            if bboxes.shape[0] == 0 or mask_preds.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type='mask',
                    instance_results=[results])[0]
            else:
                im_mask = self._predict_by_feat_single(
                    mask_preds=croped_mask_pred[img_id],
                    bboxes=bboxes,
                    img_meta=img_meta,
                    rescale=rescale)
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                img_meta: dict,
                                rescale: bool,
                                cfg: OptConfigType = None):
        """Transform a single image's features extracted from the head into
        mask results.

        Args:
            mask_preds (Tensor): Predicted prototypes, has shape [H, W, N].
            bboxes (Tensor): Bbox coords in relative point form with
                shape [N, 4].
            img_meta (dict): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If rescale is False, then returned masks will
                fit the scale of imgs[0].
            cfg (dict, optional): Config used in test phase.
                Defaults to None.

        Returns:
            :obj:`InstanceData`: Processed results of single image.
             it usually contains following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,).
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).
        """
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        # 如果需要缩放回Resize前尺寸,那么直接跳过[batch_h, batch_w],
        # 直接从[batch_h//4, batch_w//4] resize到[ori_h,ori_w]
        # 否则就resize到网络输入尺寸[batch_h, batch_w]
        if rescale:  # in-placed rescale the bboxes
            scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
                (1, 2))
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        masks = F.interpolate(
            mask_preds.unsqueeze(0), (img_h, img_w),
            mode='bilinear',
            align_corners=False).squeeze(0) > cfg.mask_thr

        if cfg.mask_thr_binary < 0:
            # for visualization and debugging
            masks = (masks * 255).to(dtype=torch.uint8)

        return masks


class SegmentationModule(BaseModule):
    """YOLACT segmentation branch used in <https://arxiv.org/abs/1904.02689>`_

    In mmdet v2.x `segm_loss` is calculated in YOLACTSegmHead, while in
    mmdet v3.x `SegmentationModule` is used to obtain the predicted semantic
    segmentation map and `segm_loss` is calculated in YOLACTProtonet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 256,
        init_cfg: ConfigType = dict(
            type='Xavier',
            distribution='uniform',
            override=dict(name='segm_conv'))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.segm_conv = nn.Conv2d(
            self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: Predicted semantic segmentation map with shape
                (N, num_classes, H, W).
        """
        return self.segm_conv(x)


class InterpolateModule(BaseModule):
    """This is a module version of F.interpolate.

    Any arguments you give it just get passed along for the ride.
    """

    def __init__(self, *args, init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Feature from the upstream network, which is
                a 4D-tensor.

        Returns:
            Tensor: A 4D-tensor feature map.
        """
        return F.interpolate(x, *self.args, **self.kwargs)
