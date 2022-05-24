# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np

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
                 rescale=True,
                 test_cfg=None,
                 loss_panoptic=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(num_things_classes, num_stuff_classes, test_cfg,
                         loss_panoptic, init_cfg, **kwargs)
        self.rescale = rescale
        self.num_proposals = num_proposals

    def forward_train(self, **kwargs):
        """MaskFormerFusionHead has no training loss."""
        return dict()

    def panoptic_postprocess(self, cls_scores, mask_preds,img_meta):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_proposal, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, H, W) for a image.

        Returns:
            np.array: Panoptic segment result of shape \
                (H, W), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        
        scores = cls_scores[:self.num_proposals][:, :self.num_things_classes]
        thing_scores, thing_labels = scores.max(dim=1)
        stuff_scores = cls_scores[
            self.num_proposals:][:, self.num_things_classes:].diag()
        stuff_labels = torch.arange(
            0, self.num_stuff_classes) + self.num_things_classes
        stuff_labels = stuff_labels.to(thing_labels.device)
        # total_masks = mask_preds
        total_masks = self.rescale_masks(mask_preds, img_meta) # should be change
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        panoptic_result = self.merge_stuff_thing(total_masks, total_labels,
                                                 total_scores)

        return panoptic_result

    def merge_stuff_thing(self,
                          total_masks,
                          total_labels,
                          total_scores):
        """get predict result.

        Args:
            total_masks (Tensor): Predict mask of each proposal.
                (num_proposal, H, W) for a image
            total_labels (Tensor): Predict labels of each proposal.
                (num_proposal,) for a image.
            total_scores (Tensor): Predict scores of each proposal.
                (num_proposal, )

        Returns:
            np.array: Panoptic segment result of shape \
                (H, W), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        H, W = total_masks.shape[-2:]
        panoptic_seg = total_masks.new_full((H, W),
                                            self.num_classes,
                                            dtype=torch.long)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_things_classes
            if isthing and total_scores[k] < self.test_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < self.test_cfg.overlap_thr:
                    continue

                panoptic_seg[mask] = total_labels[k] \
                    + current_segment_id * INSTANCE_OFFSET
                current_segment_id += 1

        return panoptic_seg.cpu().numpy()

    def semantic_postprocess(self, mask_cls, mask_pred):
        """Semantic segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_proposal, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_proposal, H, W) for a image.

        Returns:
            Tensor: Semantic segment result of shape \
                (cls_out_channels, H, W).
        """
        # TODO add semantic segmentation result
        raise NotImplementedError

    def instance_postprocess(self, mask_cls, mask_pred,img_meta):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_proposal, cls_out_channels) for a image.
            mask_pred (Tensor): Mask outputs of shape
                (num_proposal, H, W) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (N, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (N, H, W).
        """
        max_per_image = self.test_cfg.get('max_per_image', 100)
        scores_per_image, top_indices = mask_cls.flatten(0, 1).topk(
            max_per_image, sorted=True)
        mask_indices = top_indices // self.num_classes 
        labels_per_img = top_indices % self.num_classes
        masks_per_img = mask_pred[mask_indices]
        single_result = self.get_seg_masks(
                masks_per_img, labels_per_img, scores_per_image,
                self.test_cfg.mask_thr,img_meta)

        return single_result

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      mask_thr,img_meta):
        """Get predict segmentation result
        Args:
            masks_per_img (Tensor): Mask classification logits,
                shape (num_proposal, H, W).
            labels_per_img (Tensor): Mask logits, shape
                (num_proposal,).
            scores_per_img (Tensor): Classfication scores.
            mask_thr (float): Threshold of predict score.
            img_meta (dict): Information if image.
        Returns:
            bbox_result (list[numpy.array]): Predict bboxes,
                each with shape (predict_instance, 5).
            segm_result (list(tensor)): Predict instance mask,
                each with shape (H,W).         
        """
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        # seg_masks = masks_per_img
        seg_masks = seg_masks > mask_thr
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores):
        """Get predict result
        Args:
            mask_preds (Tensor): Mask classification logits,
                shape (num_proposal, H, W).
            det_labels (Tensor): Mask logits, shape
                (num_proposal,).
            cls_scores (Tensor): Classfication scores.
        Returns:
            bbox_result (list[numpy.array]): Predict bboxes,
                each with shape (predict_instance, 5).
            segm_result (list(tensor)): Predict instance mask,
                each with shape (H,W).         
        """
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bbox_result, segm_result


    def simple_test(self,
                    mask_cls_results,
                    mask_pred_results,
                    img_metas,
                    **kwargs):
        """Test segment without test-time aumengtation.

        Only the output of last decoder layers was used.

        Args:
            mask_cls_results (Tensor): Mask classification logits,
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_pred_results (Tensor): Mask logits, shape
                (batch_size, num_queries, H, W).
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
                        'pan_results': np.array, # shape = [H, W]
                        'ins_results': tuple[Tensor],
                        # semantic segmentation results are not supported yet
                        'sem_results': Tensor
                    },
                    ...
                ]
        """
        panoptic_on = self.test_cfg.get('panoptic_on', False)
        semantic_on = self.test_cfg.get('semantic_on', False)
        instance_on = self.test_cfg.get('instance_on', True)
        assert not semantic_on, 'segmantic segmentation '\
            'results are not supported yet.'

        results = []
        
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas):
            result = dict()
            if panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result,meta)
                result['pan_results'] = pan_results # numpy.arrary

            if instance_on:
                ins_results = self.instance_postprocess(
                    mask_cls_result, mask_pred_result,meta)
                result = ins_results   # TODO: there should be change

            if semantic_on:
                sem_results = self.semantic_postprocess(
                    mask_cls_result, mask_pred_result)
                result['sem_results'] = sem_results

            results.append(result)

        return results

    def rescale_masks(self, masks_per_img, img_meta):
        """rescale mask size for input shape
        Args:
            masks_per_img (Tensor): Predict mask with shape
                (num_proposal, down_sampled_H, down_sampled_W).
            img_meta (dict): List of image information.
        Return:
            seg_masks: resized mask with shape
                (num_proposal, H, W).
        """
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

