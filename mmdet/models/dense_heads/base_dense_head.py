# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32

from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """将网络输出转换为 box 结果.

        注意: 当 score_factors 不为 None时, cls_scores 通常乘以它,
        然后得到 NMS 中使用的真正scores,如 FCOS 中的 CenterNess，ATSS 中的 IoU 分支.
        在有些网络中会输出obj_score(上面的score_factors),代表了目标置信度(比如YOLO系列).
        它一般会与网络输出的cls_score相乘作为box的socre参与NMS,以及后续的结果中
        如果网络没有obj_score,则直接使用cls_score作为box的socre

        Args:
            cls_scores (list[Tensor]): 所有特征层级上预测的类别概率, shape为
                [(B, A * C, H, W),]*num_levels. 其中H、W随着层级不同而不同
            bbox_preds (list[Tensor]): 所有特征层级上预测的回归结果, shape为
                [(B, A * 4, H, W),]*num_levels.
            score_factors (list[Tensor], Optional): 所有特征层级上预测的目标概率,
                shape为[(B, A * 1, H, W),]*num_levels. 默认为None,即网络不输出该结果
            img_metas (list[dict], Optional): 图像元信息.
            cfg (mmcv.Config, Optional): 测试 / 后处理 配置,
                如果为 None, test_cfg 将被使用.
            rescale (bool): 如果为 True, 则将box*ori_h/new_h以适配原始图像的尺寸.
            with_nms (bool): 如果为 True, 在返回box之前做 nms.

        Returns:
            list[list[Tensor, Tensor]]: result_list 中的每一项都是 2 元组.
                第一项是形状为 (n, 5) 的“boxes”,其中 5 代表 (x1, y1, x2, y2, score).
                第二项是形状为 (n, ) 的“labels”.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # 比如像 Retina, FreeAnchor, Foveabox,这样的模型是没有输出obj_score的.
            with_score_factors = False
        else:
            # 比如像 FCOS, PAA, ATSS, AutoAssign,这样的模型则相反.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """将单个图像的输出转换为 predict_bbox.

        Args:
            cls_score_list (list[Tensor]): 来自单个图像的所有层级别的box score,
                每个元素shape 为 (A * C, H, W).
            bbox_pred_list (list[Tensor]): 来自单个图像的所有层级别的box回归结果
                每个元素shape 为 (A * 4, H, W).
            score_factor_list (list[Tensor]): 来自单个图像的所有层级别的box置信度
                每个元素shape (A * 1, H, W).
            mlvl_priors (list[Tensor]): 网络生成的多层级先验.在所有anchor-base的
                网络中,它的shape为(num_priors, 4). 在所有anchor-free的
                网络中,当with_stride=True时,它的shape为(num_priors, 2)
                否则它的shape仍然为(num_priors, 4).
            img_meta (dict): 图像元信息.
            cfg (mmcv.Config): 测试 / 后处理 配置,
                如果为 None, test_cfg 将被使用.
            rescale (bool): 如果为 True, 则将box*ori_h/new_h以适配原始图像的尺寸.
            with_nms (bool): 如果为 True, 对box做nms.

        Returns:
            tuple[Tensor]: 检测到的box和label的结果.
                If with_nms is False and mlvl_score_factor is None:
                    return mlvl_bboxes and mlvl_scores,
                else:
                    return mlvl_bboxes, mlvl_scores and mlvl_score_factor.
                with_nms 为 False 时一般在aug test中使用.
                如果with_nms 为 True, 则返回以下格式的数据

                - det_bboxes (Tensor): 预测 boxes, shape为[num_bboxes, 5],
                    5代表 (x1, y1, x2, y2, score).
                - det_labels (Tensor): 预测 label, shape为[num_bboxes].
        """
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):
            # 确保分类分支与回归分支的尺寸一致
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # (A*4,H,W) -> (H,W,A*4) -> (H*W*A,4)
            # 注意,这里三处的permute->reshape的操作本质都一样
            # 都是将单层特征图上的所有"box属性"都铺平,最后一个维度全是box的属性
            # bbox_pred:修正系数,score_factor:目标置信度,cls_score:分类置信度
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # 注意,从v2.0开始mmdet将前景的标签设置为 [0, num_class-1]
                # 背景id: num_class
                # 注意该种情况仅在self.use_sigmoid_cls为False的时候
                # 详情参考class AnchorHead 构造方法中
                scores = cls_score.softmax(-1)[:, :-1]

            # 在 https://github.com/open-mmlab/mmdetection/pull/6268/ 这之后
            # 此操作在相同的 `nms_pre` 下保留更少的 box.大多数模型的性能没有差异.
            # 如果你发现性能略有下降, 你可以设置比以前更大的 `nms_pre`.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """box后处理方法.

        这些框将被重新缩放到原始图像尺寸下并执行nms操作.通常 `with_nms`为False时用于aug test.

        Args:
            mlvl_scores (list[Tensor]): 来自单个图像的所有层别的box score,其中元素
                shape为(num_bboxes, ).
            mlvl_labels (list[Tensor]): 来自单个图像的所有层别的box label,其中元素
                shape为(num_bboxes, ).
            mlvl_bboxes (list[Tensor]): 来自单个图像的所有层别的box(x1 y1 x2 y2),
                其中元素shape为(num_bboxes, 4).
            scale_factor (ndarray, optional): 图像在Resize阶段的缩放因子,其格式为
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): 测试 / 后处理 配置,
                如果为 None, test_cfg 将被使用.
            rescale (bool): 如果为 True, 则将box*ori_h/new_h以适配原始图像的尺寸.
            with_nms (bool): 如果为 True, 对box做nms.
            mlvl_score_factors (list[Tensor], optional): 来自单个图像的
            所有层级的obj_score, 其中元素shape为(num_bboxes, ). 默认为None.

        Returns:
            tuple[Tensor]: 检测到的box和label的结果.
                If with_nms is False:
                    return mlvl_bboxes, mlvl_scores, mlvl_labels
                else: 则返回以下格式的数据

                - det_bboxes (Tensor): 预测 boxes shape [num_bboxes, 5],
                    5代表 (x1, y1, x2, y2, score).
                - det_labels (Tensor): 预测 label, shape为[num_bboxes].

        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): 多层级特征图[(bs, c, f_h, f_w),]*num_level.
            img_metas (list[dict]): 图像信息字典列表,其中每个字典具有:'img_shape'、
                'scale_factor'、'flip',还可能包含 'filename'、'ori_shape'、
                'pad_shape'和'img_norm_cfg'. img_metas长度为bs
                有关这些键值的详细信息, 请参见
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (Tensor): bs幅图像的gt box,其内部shape为(num_gts, 4)
                其中num_gts代表该幅图像标有num_gts个gt,4代表[x1, y1, x2, y2]
            gt_labels (Tensor): bs幅图像的gt label,其内部shape为(num_gts,)
            gt_bboxes_ignore (Tensor): 计算损失时可以忽略的指定gt box.
                shape为(num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """没有TTA的测试方法,len(TTA)=1.

        Args:
            feats (tuple[torch.Tensor]): 来自上游网络的多级特征,每个都是 4D 张量.
            img_metas (list[dict]): batch张图像信息.
            rescale (bool, optional): 是否缩放box.

        Returns:
            list[tuple[Tensor, Tensor]]: result_list 中的每一项都是 2 元组.
                第一项是形状为 (n, 5) 的“boxes”,其中 5 代表 (x1, y1, x2, y2, score).
                第二项是形状为 (n, ) 的“labels”.
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors=None,
                    img_metas=None,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            score_factors (list[Tensor]): score_factors for each s
                cale level with shape (N, num_points * 1, H, W).
                Default: None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc. Default: None.
            with_nms (bool): Whether apply nms to the bboxes. Default: True.

        Returns:
            tuple[Tensor, Tensor] | list[tuple]: When `with_nms` is True,
            it is tuple[Tensor, Tensor], first tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
            When `with_nms` is False, first tensor is bboxes with
            shape [N, num_det, 4], second tensor is raw score has
            shape  [N, num_det, num_classes].
        """
        assert len(cls_scores) == len(bbox_preds)

        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shape = img_metas[0]['img_shape_for_onnx']

        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)

        # e.g. Retina, FreeAnchor, etc.
        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
            mlvl_score_factor = [
                score_factors[i].detach() for i in range(num_levels)
            ]
            mlvl_score_factors = []

        mlvl_batch_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, score_factors, priors in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(0, 2, 3,
                                       1).reshape(batch_size, -1,
                                                  self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
                nms_pre_score = scores
            else:
                scores = scores.softmax(-1)
                nms_pre_score = scores

            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(
                    batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            priors = priors.expand(batch_size, -1, priors.size(-1))
            # Get top-k predictions
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:

                if with_score_factors:
                    nms_pre_score = (nms_pre_score * score_factors[..., None])
                else:
                    nms_pre_score = nms_pre_score

                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)

                batch_inds = torch.arange(
                    batch_size, device=bbox_pred.device).view(
                        -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                priors = priors.reshape(
                    -1, priors.size(-1))[transformed_inds, :].reshape(
                        batch_size, -1, priors.size(-1))
                bbox_pred = bbox_pred.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                scores = scores.reshape(
                    -1, self.cls_out_channels)[transformed_inds, :].reshape(
                        batch_size, -1, self.cls_out_channels)
                if with_score_factors:
                    score_factors = score_factors.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)

        batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_scores, dim=1)
        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment

        from mmdet.core.export import add_dummy_nms_for_onnx

        if not self.use_sigmoid_cls:
            batch_scores = batch_scores[..., :self.num_classes]

        if with_score_factors:
            batch_scores = batch_scores * (batch_score_factors.unsqueeze(2))

        if with_nms:
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        else:
            return batch_bboxes, batch_scores
