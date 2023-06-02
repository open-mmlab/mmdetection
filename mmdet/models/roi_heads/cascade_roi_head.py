# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from ..utils.misc import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead


<<<<<<< HEAD
@HEADS.register_module()
class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head 包括一个 bbox head 和一个 mask head.
        注意该Head模块没有shared_head mask_roi_extractor mask_head的配置
=======
@MODELS.register_module()
class CascadeRoIHead(BaseRoIHead):
    """Cascade roi head including one bbox head and one mask head.

>>>>>>> mmdetection/main
    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: Union[List[float], Tuple[float]],
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, 'Cascade-RCNN 不支持Shared head'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

<<<<<<< HEAD
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """初始化 roi box head 以及roi提取模块(RoIPooling或者RoIAlign).

        Args:
            bbox_roi_extractor (dict): roi提取模块的配置.
            bbox_head (dict): roi box head的配置.
=======
    def init_bbox_head(self, bbox_roi_extractor: MultiConfig,
                       bbox_head: MultiConfig) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of box roi extractor.
            bbox_head (:obj:`ConfigDict`, dict or list): Config
                of box in box head.
>>>>>>> mmdetection/main
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        # 下面两个if是考虑到直接在配置文件中手动修改这两项准备的,如果为list格式,
        # 则代表对这个模块有更多个性化需求需要手动修改,此时将不再重复创建
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor))
            self.bbox_head.append(MODELS.build(head))

<<<<<<< HEAD
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """初始化 mask head 以及mask roi提取模块.

        Args:
            mask_roi_extractor (dict): mask roi提取模块的配置.
            mask_head (dict): mask head的配置.
=======
    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_head (dict): Config of mask in mask head.
            mask_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of mask roi extractor.
>>>>>>> mmdetection/main
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(MODELS.build(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(MODELS.build(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

<<<<<<< HEAD
    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage.
            sample 只是pos/neg/min_iou_thr从0.5 -> 0.6 -> 0.7
            assigner则是三个完全一样的
        """
=======
    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler for each stage."""
>>>>>>> mmdetection/main
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    TASK_UTILS.build(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    TASK_UTILS.build(
                        rcnn_train_cfg.sampler,
                        default_args=dict(context=self)))

    def _bbox_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Box head forward function used in both training and testing.

<<<<<<< HEAD
    def _bbox_forward(self, stage, x, rois):
        """用于训练和测试的 Box Head前向传播函数.
            x: [[bs, c, h, w],] * num_level
            rois: 整个batch图片上的正负样本cat到一起,理论最大值为[bs*sampler.num,5]
            下面是返回数据的shape
            cls_score: [bs*sampler.num, num_class+1]
            bbox_pred: [bs*sampler.num, 4]
            bbox_feats: [bs*sampler.num, 256, 7, 7]
=======
        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
>>>>>>> mmdetection/main
        """
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        # num_inputs代表应用特征图层级的数量,rpn应用全部5个,而 r-cnn仅使用前4个
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # 不再支持 caffe_c4 相关的模型
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

<<<<<<< HEAD
    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training.
            stage: head索引
            sampling_results: bs张图片的采样结果
            x: [[bs, c, h, w],] * num_level.
        """
        # 整个batch图片上的正负样本cat到一起,[torch.cat(pos_box, neg_box),] * bs
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        # labels, label_weights, bbox_targets, bbox_weights
        # [bs * (pos_num+neg_num), ] [bs * (pos_num+neg_num), 4]
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)
=======
    def bbox_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult]) -> dict:
        """Run forward function and calculate loss for box head in training.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
        """
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_results.update(rois=rois)

        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results['cls_score'],
            bbox_pred=bbox_results['bbox_pred'],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage])
        bbox_results.update(bbox_loss_and_target)
>>>>>>> mmdetection/main

        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor],
                      rois: Tensor) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_preds = mask_head(mask_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def mask_loss(self, stage: int, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
<<<<<<< HEAD
            x (list[Tensor]): [[bs, c, h, w],] * num_level.
            img_metas (list[dict]): [{img_info},]*bs.图像信息字典列表,其中每个字典
                有'img_shape', 'scale_factor', 'flip' 等键, 并且还可能包含
                'filename', 'ori_shape', 'pad_shape', 以及 'img_norm_cfg'.
                键值详情参见`mmdet/datasets/pipelines/formatting.py:Collect`.
            proposal_list (list[Tensors]): 来自RPN的roi.
                [[rpn_proposal.max_per_img, 5], ] * bs. [x1, y1, x2, y2, score]
            gt_bboxes (list[Tensor]): [[num_gts, 4],] * bs, [x1, y1, x2, y2]
            gt_labels (list[Tensor]): [[num_gts,],] * bs
            gt_bboxes_ignore (None | list[Tensor]): 指定loss时忽略的gt box.
            gt_masks (None | Tensor) : 如果支持分割任务,则表示每个gt box的mask.
=======
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
>>>>>>> mmdetection/main

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_head = self.mask_head[stage]

        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage])
        mask_results.update(mask_loss_and_target)

        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # TODO: May add a new function in baseroihead
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            stage_loss_weight = self.stage_loss_weights[stage]

            # 分配gt box 并且对roi进行采样
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]

                for i in range(num_imgs):
                    results = results_list[i]
                    # rename rpn_results.bboxes to rpn_results.priors
                    results.priors = results.pop('bboxes')

                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                    sampling_result = bbox_sampler.sample(
                        assign_result,
<<<<<<< HEAD
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        # 指定图片上的各层级特征图 [[1,c,h,w],] * num_level
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # 与普通R-CNN仅返回loss不同,Cascade R-CNN会循环对roi回归修正,所以会额外多一些值.
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)
            """ 
            令 N = bs*(pos_num+neg_num), 在roi head中由rpn提供的roi在sampler.sample的时候
            根据add_gt_as_proposals是否将gt插入roi中,以增加roi正样本的数量与质量,然后根据
            sampler.num及相关正负样本比例参数筛选出正负样本, 并由正负样本组成新的roi返回.
            从而后续送入roi head中box head里面的RoIAlign以及FC层进行前向传播.以下是理想情况下的值.
            bbox_results = {
            "cls_score": [N, nc]
            "bbox_pred": [N, 4]
            "bbox_feats": [N, 256, 7, 7]
            "loss_bbox":{'loss_cls':tensor,'acc':tensor,'loss_bbox':tensor}
            "rois": [N, 5]
            下面的值都是基于roi初始化同长度的数据得到的,所以shape与roi对齐.
            "bbox_targets": (labels, label_weights, bbox_targets, bbox_weights)
                             [N,],    [N,],         [N, 4],       [N, 4],
            }
            """
=======
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self.bbox_loss(stage, x, sampling_results)

>>>>>>> mmdetection/main
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self.mask_loss(stage, x, sampling_results,
                                              batch_gt_instances)
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

<<<<<<< HEAD
            # 微调box, 这里相当于get_box函数
            if i < self.num_stages - 1:
                # [[num_pos, ], ] * bs. target box是否是gt,后续要把来自gt的target box剔除
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)

                    # 如果整个batch的图片上都没有roi,那么直接返回loss.
                    # 一般这种情况可能由于没有gt,或者iou阈值过高引起的
                    if cls_score.numel() == 0:
                        break
                    # 这里的代码比较难理解,但同时也比较重要.
                    # 假设某个roi与gt的iou计算过程中被判定为背景,但是经过当前head网络的修正
                    # 它是有可能被修正为正样本的,所以需要对其cls_target进行修正,
                    # 即修改为其目前cls_score(剔除背景)的最大值索引,而那些非背景类的
                    # 那些原本就是正样本,则忽略修改.
                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[i].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

=======
            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(
                        sampling_results, bbox_results, batch_img_metas)
                    # Empty proposal
                    if results_list is None:
                        break
>>>>>>> mmdetection/main
        return losses

    def predict_bbox(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
<<<<<<< HEAD
            x (tuple[Tensor]): [[bs, c, h, w],]*num_level
            proposal_list (list(Tensor)): 来自rpn的roi.
                [[num_roi, 5],] * bs, [x1, y1, x2, y2, score].
            img_metas (list[dict]): 图像的基础属性.
            rescale (bool): 是否将结果缩放回原始图像尺寸上. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: 当没有mask分支时,
            [[np.ndarray]*num_class,]*bs. 当模型含有mask分支时,
            它包含 box 和mask结果.[(box_results,mask_results)]*bs.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # 变量名中的"ms"表示多级
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        # [(sampler.num,4),]*bs -> [batch_box, 5] [batch_ind, x1, y1, x2, y2]
        rois = bbox2roi(proposal_list)

        if rois.shape[0] == 0:
            # 这里代表整个batch中没有一个roi
            bbox_results = [[
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs
=======
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None)
>>>>>>> mmdetection/main

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs)

        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg)
        return results_list

<<<<<<< HEAD
        for i in range(self.num_stages):
            # 前两个是基于rois预测出来的cls_score, reg
            # {"cls_score":[bs*sampler.num, num_class+1],
            # "bbox_pred":[bs*sampler.num, 4],
            # "bbox_feats"=[bs*sampler.num, 256, 7, 7]}
            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            # 将一个batch上的roi以图片为单位单独划分出来
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):  # sabl_head为False
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[i].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)
=======
    def predict_mask(self,
                     x: Tuple[Tensor],
                     batch_img_metas: List[dict],
                     results_list: List[InstanceData],
                     rescale: bool = False) -> List[InstanceData]:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            mask_preds = mask_results['mask_preds']
            # split batch mask prediction back to each image
            mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for i in range(len(batch_img_metas)):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)
        results_list = self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        return results_list

    def _refine_roi(self, x: Tuple[Tensor], rois: Tensor,
                    batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int], **kwargs) -> tuple:
        """Multi-stage refinement of RoI.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): shape (n, 5), [batch_ind, x1, y1, x2, y2]
            batch_img_metas (list[dict]): List of image information.
            num_proposals_per_img (sequence[int]): number of proposals
                in each image.

        Returns:
            tuple:

               - rois (Tensor): Refined RoI.
               - cls_scores (list[Tensor]): Average predicted
                   cls score per image.
               - bbox_preds (list[Tensor]): Bbox branch predictions
                   for the last stage of per image.
        """
        # "ms" in variable names means multi-stage
        ms_scores = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage=stage, x=x, rois=rois, **kwargs)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score']
            bbox_preds = bbox_results['bbox_pred']
>>>>>>> mmdetection/main

            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            # some detector with_reg is False, bbox_preds will be None
            if bbox_preds is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head[stage].bbox_pred_split(
                        bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None, ) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                if bbox_head.custom_activation:
                    cls_scores = [
                        bbox_head.loss_cls.get_activation(s)
                        for s in cls_scores
                    ]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(dim=1)
                        # Refactor `bbox_head.regress_by_class` to only accept
                        # box tensor without img_idx concatenated.
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i],
                            batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)
                        refined_rois = torch.cat(
                            [rois[i][:, [0]], refined_bboxes], dim=1)
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

<<<<<<< HEAD
        # 最终的cls_score不是最后一个stage输出,而是所有stage的平均值
        cls_score = [
=======
        # average scores of each image by stages
        cls_scores = [
>>>>>>> mmdetection/main
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(len(batch_img_metas))
        ]
        return rois, cls_scores, bbox_preds

<<<<<<< HEAD
        # 分别对每个图像使用box后处理
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
=======
    def forward(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
>>>>>>> mmdetection/main

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

<<<<<<< HEAD
        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
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
                mask_rois = bbox2roi(_bboxes)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)
                aug_masks = []
                for i in range(self.num_stages):
                    mask_results = self._mask_forward(i, x, mask_rois)
                    mask_pred = mask_results['mask_pred']
                    # 将mask分支的整个batch预测结果分割成以图片为单位的列表结果
                    mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                    aug_masks.append([
                        m.sigmoid().cpu().detach().numpy() for m in mask_pred
                    ])

                # 分别对每个图像应用mask后处理
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        aug_mask = [mask[i] for mask in aug_masks]
                        merged_masks = merge_aug_masks(
                            aug_mask, [[img_metas[i]]] * self.num_stages,
                            rcnn_test_cfg)
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:  # 将每张图片的box结果与mask结果整合到一起
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
=======
        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
>>>>>>> mmdetection/main
        """
        results = ()
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        # bbox head
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(
                x, rois, batch_img_metas, num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        # mask head
        if self.with_mask:
            aug_masks = []
            rois = torch.cat(rois)
            for stage in range(self.num_stages):
                mask_results = self._mask_forward(stage, x, rois)
                mask_preds = mask_results['mask_preds']
                mask_preds = mask_preds.split(num_proposals_per_img, 0)
                aug_masks.append([m.sigmoid().detach() for m in mask_preds])

            merged_masks = []
            for i in range(len(batch_img_metas)):
                aug_mask = [mask[i] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
                merged_masks.append(merged_mask)
            results = results + (merged_masks, )
        return results
