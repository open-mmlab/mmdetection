# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
from mmdet.core.mask import mask2bbox
from mmdet.models.builder import HEADS
from .base_panoptic_fusion_head import BasePanopticFusionHead


@HEADS.register_module()
class KNetFusionHead(BasePanopticFusionHead):

    def __init__(self,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_proposals=100,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            test_cfg=test_cfg,
            loss_panoptic=loss_panoptic,
            init_cfg=init_cfg,
            **kwargs)
        self.num_proposals = num_proposals

    def forward_train(self, **kwargs):
        """KNetFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_proposals + num_stuff_classes, cls_out_channels) for a
                image. Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_proposals + num_stuff_classes, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        thing_scores = mask_cls[:self.num_proposals][:, :self.
                                                     num_things_classes]
        thing_scores, thing_labels = thing_scores.max(dim=1)

        stuff_scores = mask_cls[
            self.num_proposals:][:, self.num_things_classes:].diag()
        stuff_labels = torch.arange(
            0, self.num_stuff_classes) + self.num_things_classes
        stuff_labels = stuff_labels.to(thing_labels.device)

        scores = torch.cat([thing_scores, stuff_scores], dim=0)
        labels = torch.cat([thing_labels, stuff_labels], dim=0)

        h, w = mask_pred.shape[-2:]
        panoptic_seg = mask_pred.new_full((h, w),
                                          self.num_classes,
                                          dtype=torch.long)
        cur_pro_masks = scores.view(-1, 1, 1) * mask_pred
        cur_mask_ids = cur_pro_masks.argmax(0)

        sorted_inds = torch.argsort(-scores)
        segment_id = 0
        for k in sorted_inds:
            label = labels[k].item()
            isthing = label < self.num_things_classes
            if isthing and scores[k] < self.test_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (mask_pred[k] >= 0.5).sum().item()
            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < self.test_cfg.overlap_thr:
                    continue

                panoptic_seg[mask] = label + segment_id * INSTANCE_OFFSET
                segment_id += 1

        return panoptic_seg

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_proposals, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_proposals, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels (Tensor): Predicted labels, shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of
              positive region in binary mask, the last column is scores.
            - masks (Tensor): Instance masks of shape (n, h, w).
        """
        max_per_img = self.test_cfg.get('max_per_img', 100)
        scores, topk_indices = mask_cls.flatten(0, 1).topk(
            max_per_img, sorted=True)
        mask_indices = topk_indices // self.num_classes
        labels = topk_indices % self.num_classes
        masks = mask_pred[mask_indices]
        masks = masks > self.test_cfg.mask_thr
        bboxes = mask2bbox(masks)
        bboxes = torch.cat([bboxes, scores[:, None]], dim=-1)

        return labels, bboxes, masks

    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, n, cls_out_channels).
                Note `cls_out_channels` should includes background.
                n is (num_proposals + num_stuff) for panoptic segmentation,
                num_proposals for instance segmentation.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, n, h, w). n is (num_proposals + num_stuff)
                for panoptic segmentation, num_proposals for instance
                segmentation.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): If True, return boxes in
                original image space. Default False.

        Returns:
            list[dict[str, Tensor | tuple[Tensor]]]: Semantic segmentation \
                results and panoptic segmentation results for each \
                image.

            .. code-block:: none

                [
                    {
                        'pan_results': Tensor, # shape = [h, w]
                        'ins_results': tuple[Tensor],
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        instance_on = self.test_cfg.get('instance_on', False)

        assert (panoptic_on and not instance_on) or (not panoptic_on
                                                     and instance_on), \
            'instance segmentation and panoptic segmentation ' \
            'can\'t be turn on at the same time'

        results = []
        for mask_cls_result, mask_pred_result, img_meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            # resize to batch_input_shape
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=img_meta['batch_input_shape'],
                mode='bilinear',
                align_corners=False)[:, 0]

            # remove padding
            img_height, img_width = img_meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                ori_height, ori_width = img_meta['ori_shape'][:2]
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['pan_results'] = pan_results

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                result['ins_results'] = ins_results

            results.append(result)

        return results
