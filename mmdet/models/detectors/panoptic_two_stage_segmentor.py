# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple

import torch
from mmengine.data import PixelData
from torch import Tensor

from mmdet.core import DetDataSample, bbox2roi, multiclass_nms
from mmdet.core.utils import (ConfigType, OptConfigType, OptMultiConfig,
                              SampleList)
from mmdet.registry import MODELS
from ..roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from .two_stage import TwoStageDetector


@MODELS.register_module()
class TwoStagePanopticSegmentor(TwoStageDetector):
    """Base class of Two-stage Panoptic Segmentor.

    As well as the components in TwoStageDetector, Panoptic Segmentor has extra
    semantic_head and panoptic_fusion_head.
    """

    def __init__(
            self,
            backbone: ConfigType,
            neck: OptConfigType = None,
            rpn_head: OptConfigType = None,
            roi_head: OptConfigType = None,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            # for panoptic segmentation
            semantic_head: OptConfigType = None,
            panoptic_fusion_head: OptConfigType = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if semantic_head is not None:
            self.semantic_head = MODELS.build(semantic_head)

        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

            self.num_things_classes = self.panoptic_fusion_head.\
                num_things_classes
            self.num_stuff_classes = self.panoptic_fusion_head.\
                num_stuff_classes
            self.num_classes = self.panoptic_fusion_head.num_classes

    @property
    def with_semantic_head(self) -> bool:
        """bool: whether the detector has semantic head"""
        return hasattr(self,
                       'semantic_head') and self.semantic_head is not None

    @property
    def with_panoptic_fusion_head(self) -> bool:
        """bool: whether the detector has panoptic fusion head"""
        return hasattr(self, 'panoptic_fusion_head') and \
            self.panoptic_fusion_head is not None

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Not support currently, should have a check at Fast R-CNN
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples, **kwargs)
        losses.update(roi_losses)

        semantic_loss = self.semantic_head.loss(x, batch_data_samples,
                                                **kwargs)
        losses.update(semantic_loss)

        return losses

    def _predict_mask(self,
                      x: Tuple[Tensor],
                      batch_img_metas: List[dict],
                      det_bboxes: List[Tensor],
                      det_labels: List[Tensor],
                      rescale: bool = False) -> Dict[str, Tensor]:
        """Simple test for mask head without augmentation.

        Args:
            x (Tuple[Tensor]): Tuple of multi-level img features.
            batch_img_metas (List[dict]): List of image information.
            det_bboxes (List[Tensor]): Detected bboxes of each image.
            det_labels (List[Tensor]): Labels of the Detected bboxes of
                each image.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            Dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_pred` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
                - `masks` (List[Tensor]): Instance masks.
        """
        img_shapes = tuple(meta['ori_shape']
                           for meta in batch_img_metas) if rescale else tuple(
                               meta['batch_input_shape']
                               for meta in batch_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in batch_img_metas)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            masks = []
            for img_shape in img_shapes:
                out_shape = (0, self.roi_head.bbox_head.num_classes) \
                    + img_shape[:2]
                masks.append(det_bboxes[0].new_zeros(out_shape))
            mask_pred = det_bboxes[0].new_zeros((0, 80, 28, 28))
            mask_results = dict(
                masks=masks, mask_pred=mask_pred, mask_feats=None)
            return mask_results

        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = [
                    det_bboxes[0].new_tensor(scale_factor).repeat((1, 2))
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                _bboxes[i] * scale_factors[i] for i in range(len(_bboxes))
            ]

        mask_rois = bbox2roi(_bboxes)
        mask_results = self.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        # resize the mask_preds to (K, H, W)
        masks = []
        for i in range(len(_bboxes)):
            det_bbox = det_bboxes[i][:, :4]
            det_label = det_labels[i]

            mask_pred = mask_preds[i].sigmoid()

            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, det_label][:, None]

            img_h, img_w = img_shapes[i][:2]
            mask_pred, _ = _do_paste_mask(
                mask_pred, det_bbox, img_h, img_w, skip_empty=False)
            masks.append(mask_pred)

        mask_results['masks'] = masks

        return mask_results

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            List[:obj:`DetDataSample`]: Return the packed panoptic segmentation
                results of input images. Each DetDataSample usually contains
                'pred_panoptic_seg'. And the 'pred_panoptic_seg' has a key
                ``sem_seg``, which is a tensor of shape (1, h, w).
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # instance data
        roi_results_list = self.roi_head.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=None,
            rescale=rescale)

        seg_preds = self.semantic_head.predict(x, batch_img_metas, rescale)

        pan_cfg = self.test_cfg.panoptic
        # class-wise predictions
        det_bboxes = []
        det_labels = []
        for i in range(len(roi_results_list)):
            bboxe = roi_results_list[i]['bboxes']
            score = roi_results_list[i]['scores']
            det_bbox, det_label = multiclass_nms(bboxe, score,
                                                 pan_cfg.score_thr,
                                                 pan_cfg.nms,
                                                 pan_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        mask_results = self._predict_mask(
            x, batch_img_metas, det_bboxes, det_labels, rescale=rescale)
        masks = mask_results['masks']

        results_list = self.panoptic_fusion_head.predict(
            det_bboxes, det_labels, masks, seg_preds)

        results_list = self.convert_to_datasample(results_list)

        return results_list

    # TODO the code has not been verified and needs to be refactored later.
    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList,
                 **kwargs) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head``, ``roi_head`` and
                ``semantic_head`` forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)
        rpn_outs = self.rpn_head.forward(x)
        results = results + (rpn_outs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            rpn_results_list = self.rpn_head.predict_by_feat(
                *rpn_outs, batch_img_metas=batch_img_metas, rescale=False)
        else:
            # TODO: Not checked currently.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # roi_head
        roi_outs = self.roi_head._forward(x, rpn_results_list)
        results = results + (roi_outs)

        # semantic_head
        sem_outs = self.semantic_head.forward(x)
        results = results + (sem_outs['seg_preds'], )

        return results

    def convert_to_datasample(self,
                              results_list: List[PixelData]) -> SampleList:
        """Convert results list to `DetDataSample`.

        Args:
            results_list (List[PixelData]): Panoptic segmentation results of
                each image.

        Returns:
            List[:obj:`DetDataSample`]: Return the packed panoptic segmentation
                results of input images. Each DetDataSample usually contains
                'pred_panoptic_seg'. And the 'pred_panoptic_seg' has a key
                ``sem_seg``, which is a tensor of shape (1, h, w).
        """
        results = []
        for pred_panoptic_seg in results_list:
            result = DetDataSample()
            result.pred_panoptic_seg = pred_panoptic_seg
            results.append(result)

        return results
