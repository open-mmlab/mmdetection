# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList, OptMultiConfig
from ..test_time_augs import merge_aug_results
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads.

    1. The ``init_weights`` method is used to initialize densehead's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of densehead,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call densehead's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the densehead needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.

    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # `_raw_positive_infos` will be used in `get_positive_infos`, which
        # can get positive information.
        self._raw_positive_infos = dict()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get(
            'sampling_results', None)
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        pass

<<<<<<< HEAD
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
=======
    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
>>>>>>> mmdetection/main

        注意: 当 score_factors 不为 None时, cls_scores 通常乘以它,
        然后得到 NMS 中使用的真正scores,如 FCOS 中的 CenterNess, ATSS 中的 IoU 分支.
        在有些网络中会输出obj_score(上面的score_factors),代表了目标置信度(比如YOLO系列).
        它一般会与网络输出的cls_score相乘作为box的socre参与NMS,以及后续的结果中
        如果网络没有obj_score,则直接使用cls_score作为box的socre

        Args:
<<<<<<< HEAD
            cls_scores (list[Tensor]): 所有特征层级上cls score,
                [[bs, na * nc, h, w],]*num_levels. 其中H、W随着层级不同而不同
            bbox_preds (list[Tensor]): 所有特征层级上box reg,
                [[bs, na * 4, h, w],]*num_levels.
            score_factors (list[Tensor], Optional): 所有特征层级上box conf,
                [[bs, na, h, w], ]*num_levels. 默认为None,即网络不输出该结果
            img_metas (list[dict], Optional): [dict] * bs,图像元信息.
            cfg (mmcv.Config, Optional): 测试 / 后处理 配置,
                如果为 None, test_cfg 将被使用.
            rescale (bool): 如果为 True, 则将box*ori_h/new_h以适配原始图像的尺寸.
            with_nms (bool): 如果为 True, 在返回box之前做 nms.

        Returns:
            list[list[Tensor, Tensor]]: [[det_box, det_label], ] * bs.
                det_box: [n, 5],其中 5 代表 [x1, y1, x2, y2, score].
                det_label: [n,].
=======
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
>>>>>>> mmdetection/main
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

<<<<<<< HEAD
        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            # [[na*nc, h, w],] * num_level. 其他同理
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
=======
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
>>>>>>> mmdetection/main
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

<<<<<<< HEAD
            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            # 默认[det_bboxes[n,5], det_labels[n,]], RPN这种网络时仅有det_bboxes
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
            cls_score_list (list[Tensor]): 来自单个图像的所有层级别的cls score,
                [[na * nc, h, w],] * num_level.
            bbox_pred_list (list[Tensor]): 来自单个图像的所有层级别的box reg
                [[na * 4, h, w],] * num_level.
            score_factor_list (list[Tensor]): 来自单个图像的所有层级别的box conf
                [[na, h, w],] * num_level.
            mlvl_priors (list[Tensor]): 网络生成的多层级先验.
                在anchor-base中,[[h * w * na, 4], ] * num_level. na代表num_anchor.
                在anchor-free中,[[h * w * np, 2], ] * num_level. np代表num_prior.
                但当with_stride=True时, 最后一维为4->[x, y, stride_w, stride_h]
            img_meta (dict): 图像元信息.
            cfg (mmcv.Config): 测试 或者 后处理 配置,
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

                - det_bboxes (Tensor): [num_bboxes, 5],
                - det_labels (Tensor): [num_bboxes].
=======
            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
>>>>>>> mmdetection/main
        """
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
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
<<<<<<< HEAD
            # (na*4,h,w) -> (h,w,na*4) -> (h*w*na,4)
            # 注意,这里三处的permute->reshape的操作本质都一样
            # 都是将单层特征图上的所有"box属性"都铺平,最后一个维度全是box的属性
            # bbox_pred:修正系数,score_factor:目标置信度,cls_score:分类置信度
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
=======

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
>>>>>>> mmdetection/main
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,  # [h * w * na, nc]
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # 前景∈ [0, num_class),背景: num_class
                # 注意该种情况仅在self.use_sigmoid_cls为False的时候
                # 详情参考class AnchorHead 构造方法中
                scores = cls_score.softmax(-1)[:, :-1]

<<<<<<< HEAD
            # 在 https://github.com/open-mmlab/mmdetection/pull/6268/ 这之后
            # 此操作在相同的 `nms_pre` 下保留更少的 box.大多数模型的性能没有差异.
            # 如果你发现性能略有下降, 你可以设置比以前更大的 `nms_pre`.
=======
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

>>>>>>> mmdetection/main
            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            # 注意此处的score已被展平为一维数据,
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']


            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
<<<<<<< HEAD
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
=======
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.
>>>>>>> mmdetection/main

        这些框将被重新缩放到原始图像尺寸下并执行nms操作.通常 `with_nms`为False时用于aug test.

        Args:
<<<<<<< HEAD
            mlvl_scores (list[Tensor]): 单个图像上所有层级的box score
                [[num_bboxes, nc], ] * num_level
            mlvl_labels (list[Tensor]): 单个图像上所有层级的box label
                [[num_bboxes, ], ] * num_level
            mlvl_bboxes (list[Tensor]): 单个图像上所有层级的box
                [[num_bboxes, 4], ] * num_level
            scale_factor (ndarray, optional): 图像在Resize阶段的缩放因子,其格式为
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): 测试 / 后处理 配置,
                如果为 None, test_cfg 将被使用.
            rescale (bool): 如果为 True, 则将box*ori_h/new_h以适配原始图像的尺寸.
            with_nms (bool): 如果为 True, 对box做nms.
            mlvl_score_factors (list[Tensor], optional): 单个图像上所有层级的obj_score
                [[num_bboxes, ], ] * num_level.

        Returns:
            tuple[Tensor]: 检测到的box和label的结果.
                If with_nms is False:
                    return mlvl_bboxes, mlvl_scores, mlvl_labels
                else: 则返回以下格式的数据

                - det_bboxes (Tensor): [num_bboxes, 5]
                - det_labels (Tensor): [num_bboxes,]

        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)
        # 将单张图像上不同层级之间的 box/score/labels 合并到一起
        mlvl_bboxes = torch.cat(mlvl_bboxes)
=======
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
>>>>>>> mmdetection/main
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

<<<<<<< HEAD
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
            x (list[Tensor]): 多层级特征图[[bs, c, h, w],]*num_level.
            img_metas (list[dict]): 图像信息字典列表,其中每个字典具有:'img_shape'、
                'scale_factor'、'flip',还可能包含 'filename'、'ori_shape'、
                'pad_shape'和'img_norm_cfg'. img_metas长度为bs
                有关这些键值的详细信息, 请参见
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (Tensor): [[num_gts, 4],] * bs
            gt_labels (Tensor): [[num_gts,],] * bs
            gt_bboxes_ignore (Tensor): 计算损失时可以忽略的指定gt box.
                shape为(num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): 测试/后处理配置,
                为None时, 将使用test_cfg

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
            feats (tuple[torch.Tensor]): [[bs, c, h, w],] * num_level.
            img_metas (list[dict]): batch张图像信息.
            rescale (bool, optional): 是否缩放box.

        Returns:
            list[tuple[Tensor, Tensor]]: [[det_box, det_label], ] * bs.
                det_box: [n, 5],其中 5 代表 (x1, y1, x2, y2, score).
                det_label: det_box对应的label, [n,].
=======
        return results

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[tuple[Tensor]]): The outer list
                indicates test-time augmentations and inner tuple
                indicate the multi-level feats from
                FPN, each Tensor should have a shape (B, C, H, W),
            aug_batch_img_metas (list[list[dict]]): Meta information
                of images under the different test-time augs
                (multiscale, flip, etc.). The outer list indicate
                the
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            with_ori_nms (bool): Whether execute the nms in original head.
                Defaults to False. It will be `True` when the head is
                adopted as `rpn_head`.

        Returns:
            list(obj:`InstanceData`): Detection results of the
            input images. Each item usually contains\
            following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances,).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
>>>>>>> mmdetection/main
        """
        # TODO: remove this for detr and deformdetr
        sig_of_get_results = signature(self.get_results)
        get_results_args = [
            p.name for p in sig_of_get_results.parameters.values()
        ]
        get_results_single_sig = signature(self._get_results_single)
        get_results_single_sig_args = [
            p.name for p in get_results_single_sig.parameters.values()
        ]
        assert ('with_nms' in get_results_args) and \
               ('with_nms' in get_results_single_sig_args), \
               f'{self.__class__.__name__}' \
               'does not support test-time augmentation '

        num_imgs = len(aug_batch_img_metas[0])
        aug_batch_results = []
        for x, img_metas in zip(aug_batch_feats, aug_batch_img_metas):
            outs = self.forward(x)
            batch_instance_results = self.get_results(
                *outs,
                img_metas=img_metas,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=with_ori_nms,
                **kwargs)
            aug_batch_results.append(batch_instance_results)

        # after merging, bboxes will be rescaled to the original image
        batch_results = merge_aug_results(aug_batch_results,
                                          aug_batch_img_metas)

        final_results = []
        for img_id in range(num_imgs):
            results = batch_results[img_id]
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            # some nms operation may reweight the score such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:self.test_cfg.max_per_img]
            if rescale:
                # all results have been mapped to the original scale
                # in `merge_aug_results`, so just pass
                pass
            else:
                # map to the first aug image scale
                scale_factor = results.bboxes.new_tensor(
                    aug_batch_img_metas[0][img_id]['scale_factor'])
                results.bboxes = \
                    results.bboxes * scale_factor

            final_results.append(results)

        return final_results
