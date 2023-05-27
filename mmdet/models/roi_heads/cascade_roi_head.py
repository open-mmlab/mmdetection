# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Cascade roi head 包括一个 bbox head 和一个 mask head.
        注意该Head模块没有shared_head mask_roi_extractor mask_head的配置
    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, 'Cascade-RCNN 不支持Shared head'

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super(CascadeRoIHead, self).__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """初始化 roi box head 以及roi提取模块(RoIPooling或者RoIAlign).

        Args:
            bbox_roi_extractor (dict): roi提取模块的配置.
            bbox_head (dict): roi box head的配置.
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
            self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
            self.bbox_head.append(build_head(head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """初始化 mask head 以及mask roi提取模块.

        Args:
            mask_roi_extractor (dict): mask roi提取模块的配置.
            mask_head (dict): mask head的配置.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(
                    build_roi_extractor(roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage.
            sample 只是pos/neg/min_iou_thr从0.5 -> 0.6 -> 0.7
            assigner则是三个完全一样的
        """
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                outs = outs + (mask_results['mask_pred'], )
        return outs

    def _bbox_forward(self, stage, x, rois):
        """用于训练和测试的 Box Head前向传播函数.
            x: [[bs, c, h, w],] * num_level
            rois: 整个batch图片上的正负样本cat到一起,理论最大值为[bs*sampler.num,5]
            下面是返回数据的shape
            cls_score: [bs*sampler.num, num_class+1]
            bbox_pred: [bs*sampler.num, 4]
            bbox_feats: [bs*sampler.num, 256, 7, 7]
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

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred)
        return mask_results

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # 分配gt box 并且对roi进行采样
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
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
            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (
                        value * lw if 'loss' in name else value)

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

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
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

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

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

            if i < self.num_stages - 1:
                if self.bbox_head[i].custom_activation:
                    cls_score = [
                        self.bbox_head[i].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[i].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # 最终的cls_score不是最后一个stage输出,而是所有stage的平均值
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

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

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results

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
        """
        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(features, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])

            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, rois)
                ms_scores.append(bbox_results['cls_score'])

                if i < self.num_stages - 1:
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head[-1].num_classes)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta in zip(features, img_metas):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    flip_direction = img_meta[0]['flip_direction']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip, flip_direction)
                    mask_rois = bbox2roi([_bboxes])
                    for i in range(self.num_stages):
                        mask_results = self._mask_forward(i, x, mask_rois)
                        aug_masks.append(
                            mask_results['mask_pred'].sigmoid().cpu().numpy())
                        aug_img_metas.append(img_meta)
                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg)

                ori_shape = img_metas[0][0]['ori_shape']
                dummy_scale_factor = np.ones(4)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=dummy_scale_factor,
                    rescale=False)
            return [(bbox_result, segm_result)]
        else:
            return [bbox_result]

    def onnx_export(self, x, proposals, img_metas):

        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals.shape[0] == 1, 'Only support one input image ' \
                                        'while in exporting to ONNX'
        # remove the scores
        rois = proposals[..., :-1]
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]
        # Eliminate the batch dimension
        rois = rois.view(-1, 4)

        # add dummy batch index
        rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois], dim=-1)

        max_shape = img_metas[0]['img_shape_for_onnx']
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        for i in range(self.num_stages):
            bbox_results = self._bbox_forward(i, x, rois)

            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            # Recover the batch dimension
            rois = rois.reshape(batch_size, num_proposals_per_img,
                                rois.size(-1))
            cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                          cls_score.size(-1))
            bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
            ms_scores.append(cls_score)
            if i < self.num_stages - 1:
                assert self.bbox_head[i].reg_class_agnostic
                new_rois = self.bbox_head[i].bbox_coder.decode(
                    rois[..., 1:], bbox_pred, max_shape=max_shape)
                rois = new_rois.reshape(-1, new_rois.shape[-1])
                # add dummy batch index
                rois = torch.cat([rois.new_zeros(rois.shape[0], 1), rois],
                                 dim=-1)

        cls_score = sum(ms_scores) / float(len(ms_scores))
        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img, 4)
        rois = rois.reshape(batch_size, num_proposals_per_img, -1)
        det_bboxes, det_labels = self.bbox_head[-1].onnx_export(
            rois, cls_score, bbox_pred, max_shape, cfg=rcnn_test_cfg)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            batch_index = torch.arange(
                det_bboxes.size(0),
                device=det_bboxes.device).float().view(-1, 1, 1).expand(
                    det_bboxes.size(0), det_bboxes.size(1), 1)
            rois = det_bboxes[..., :4]
            mask_rois = torch.cat([batch_index, rois], dim=-1)
            mask_rois = mask_rois.view(-1, 5)
            aug_masks = []
            for i in range(self.num_stages):
                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                aug_masks.append(mask_pred)
            max_shape = img_metas[0]['img_shape_for_onnx']
            # calculate the mean of masks from several stage
            mask_pred = sum(aug_masks) / len(aug_masks)
            segm_results = self.mask_head[-1].onnx_export(
                mask_pred, rois.reshape(-1, 4), det_labels.reshape(-1),
                self.test_cfg, max_shape)
            segm_results = segm_results.reshape(batch_size,
                                                det_bboxes.shape[1],
                                                max_shape[0], max_shape[1])
            return det_bboxes, det_labels, segm_results
