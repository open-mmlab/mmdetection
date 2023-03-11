# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.evaluation.functional import INSTANCE_OFFSET
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.mask import mask2bbox
from mmdet.utils import OptConfigType, OptMultiConfig
from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead
from mmdet.utils.memory import AvoidCUDAOOM


@MODELS.register_module()
class MaskDINOFusionHead(MaskFormerFusionHead):
    """MaskDINO fusion head which postprocesses results for panoptic
    segmentation, instance segmentation and semantic segmentation."""

    def __init__(self,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 test_cfg: OptConfigType = None,
                 loss_panoptic: OptConfigType = None,  # TODO: not used ?
                 init_cfg: OptMultiConfig = None,
                 **kwargs):
        super().__init__(
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            test_cfg=test_cfg,
            loss_panoptic=loss_panoptic,
            init_cfg=init_cfg,
            **kwargs)

    def predict(self,
                mask_cls_results: Tensor,
                mask_pred_results: Tensor,
                mask_box_results: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                **kwargs) -> List[dict]:
        """Test segment without test-time aumengtation.
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        panoptic_on = self.test_cfg.get('panoptic_on', True)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', False)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        for mask_cls_result, mask_pred_result, mask_box_result, meta in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batch_img_metas):
            # shape of image before pipeline
            ori_height, ori_width = meta['ori_shape'][:2]
            # shape of image after pipeline and before padding divisibly
            img_height, img_width = meta['img_shape'][:2]
            # shape of image after padding divisibly
            batch_input_height, batch_input_width = meta['batch_input_shape'][:2]

            # remove padding
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]

            if rescale:
                # return result in original resolution
                mask_pred_result = F.interpolate(
                    mask_pred_result[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]

            result = dict()
            if panoptic_on:
                result['pan_results'] = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)

            if instance_on:
                mask_box_results = bbox_cxcywh_to_xyxy(mask_box_results)
                height_factor = batch_input_height / img_height * ori_height
                width_factor = batch_input_width / img_width * ori_width
                mask_box_results[:, 0::2] = mask_box_results[:, 0::2] * width_factor
                mask_box_results[:, 1::2] = mask_box_results[:, 1::2] * height_factor
                result['ins_results'] = self.instance_postprocess(
                    mask_cls_result, mask_pred_result, mask_box_result)

            if semantic_on:
                result['sem_results'] = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)

            results.append(result)

        return results

    @AvoidCUDAOOM.retry_if_cuda_oom  # TODO: why MaskFormerFusionHead not use?
    def panoptic_postprocess(self, mask_cls: Tensor,
                             mask_pred: Tensor) -> PixelData:
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            :obj:`PixelData`: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        test_cfg = self.test_cfg.panoptic_postprocess_cfg
        object_mask_thr = test_cfg.get('object_mask_thr', 0.25)  # 0.8 for mask2former
        iou_thr = test_cfg.get('iou_thr', 0.8)
        filter_low_score = test_cfg.get('filter_low_score', False)
        panoptic_temperature = test_cfg.get('panoptic_temperature', 0.06)  # TODO: difference
        transform_eval = test_cfg.get('transform_eval', True)  # TODO: difference

        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.  # TODO: difference
        scores, labels = mask_cls.sigmoid().max(-1)  # TODO: difference
        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)

        # added process
        if transform_eval:  # TODO: difference
            scores, labels = F.softmax(mask_cls.sigmoid() / panoptic_temperature, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks  # TODO： ？

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)  # TODO： ？

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:  # TODO： ？
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1

        return PixelData(sem_seg=panoptic_seg[None])

    @AvoidCUDAOOM.retry_if_cuda_oom
    def instance_postprocess(self, mask_cls: Tensor,
                             mask_pred: Tensor, mask_box: Optional[Tensor]) -> InstanceData:
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.
            mask_box (Tensor): TODO

        Returns:
            :obj:`InstanceData`: Instance segmentation results.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # TODO: merge into MaskFormerFusionHead
        max_per_image = self.test_cfg.get('max_per_image', 100)
        focus_on_box = self.test_cfg.get('focus_on_box', False)

        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = mask_cls.sigmoid()  # TODO: modify MaskFormerFusionHead to add an arg use_sigmoid  # TODO: difference
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)  # TODO：why ？
        # shape (num_queries * num_class)
        labels = torch.arange(self.num_classes, device=mask_cls.device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)  # TODO：why ？
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes  # TODO：why ？
        mask_pred = mask_pred[query_indices]
        mask_box = mask_box[query_indices] if mask_box is not None else None  # TODO: difference

        # extract things  # TODO： if self.panoptic_on ?
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]
        mask_box = mask_box[is_thing] if mask_box is not None else None  # TODO: difference

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)  # TODO：why ？
        det_scores = scores_per_image * mask_scores_per_image  # TODO：why ？
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask_box if mask_box is not None else mask2bbox(mask_pred_binary)  # TODO: difference

        results = InstanceData()
        results.bboxes = bboxes
        results.labels = labels_per_image
        results.scores = det_scores if not focus_on_box else 1.0
        results.masks = mask_pred_binary
        return results
