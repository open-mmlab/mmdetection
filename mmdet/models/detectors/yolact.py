# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_head
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLACT(SingleStageDetector):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLACT, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, init_cfg)
        self.segm_head = build_head(segm_head)
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        feat = self.extract_feat(img)
        bbox_outs = self.bbox_head(feat)
        prototypes = self.mask_head.forward_dummy(feat[0])
        return (bbox_outs, prototypes)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            img (Tensor): of shape (bs, c, H, W) 输入数据.
                通常这些数据应该是经过归一化的(减均值除以方差).
            img_metas (list[dict]): 图像信息字典列表,其中每一个字典 含有键: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                有关这些键值的详细信息, 请参见
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): batch张图片的gt box, [[num_gts, 4],] * bs
                其中4代表 [x1, y1, x2, y2].
            gt_labels (list[Tensor]): gt box对应的label, [[num_gts,],] * bs
            gt_bboxes_ignore (None | list[Tensor]): 计算损失时可以忽略的标注类别列表.
            gt_masks (None | Tensor) : 如变量名.它是gt box的mask
                其类型结构为[BitmapMasks(num_masks=num_gt, height=550, width=550),] * bs

        Returns:
            dict[str, Tensor]: 模型的loss字典
        """
        # 将Bitmap mask或Polygon Mask转换为Tensor [[num_gt, H, W],] * bs
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.uint8, device=img.device)
            for gt_mask in gt_masks
        ]

        x = self.extract_feat(img)

        cls_score, bbox_pred, coeff_pred = self.bbox_head(x)
        bbox_head_loss_inputs = (cls_score, bbox_pred) + (gt_bboxes, gt_labels,
                                                          img_metas)
        losses, sampling_results = self.bbox_head.loss(
            *bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)
        losses.update(loss_segm)

        mask_pred = self.mask_head(x[0], coeff_pred, gt_bboxes, img_metas,
                                   sampling_results)
        loss_mask = self.mask_head.loss(mask_pred, gt_masks, gt_bboxes,
                                        img_metas, sampling_results)
        losses.update(loss_mask)

        # 检查是否存在INF和NaN
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{}中存在INF或NaN!'\
                .format(loss_name)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """非TTA模式."""
        feat = self.extract_feat(img)
        det_bboxes, det_labels, det_coeffs = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        segm_results = self.mask_head.simple_test(
            feat,
            det_bboxes,
            det_labels,
            det_coeffs,
            img_metas,
            rescale=rescale)

        return list(zip(bbox_results, segm_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """TTA."""
        raise NotImplementedError(
            'YOLACT 不支持 TTA')
