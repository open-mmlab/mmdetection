# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import DetDataSample, bbox2result
from mmdet.registry import MODELS
from .base import BaseDetector


@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 preprocess_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(
            preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs):
        """Directly extract features from the backbone+neck."""
        feats = self.backbone(batch_inputs)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def forward_dummy(self, batch_inputs):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        feats = self.extract_feat(batch_inputs)
        outs = self.bbox_head(feats)
        return outs

    def forward_train(self, batch_inputs, batch_data_samples, **kwargs):
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(batch_inputs,
                                                       batch_data_samples)
        feats = self.extract_feat(batch_inputs)
        losses = self.bbox_head.forward_train(feats, batch_data_samples,
                                              **kwargs)
        return losses

    def simple_test(self, batch_inputs, batch_img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            batch_inputs (torch.Tensor): Inputs with shape (N, C, H, W).
            batch_img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the \
                input images. Each DetDataSample usually contain \
                'pred_instances'. And the ``pred_instances`` usually \
                contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        feats = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.simple_test(
            feats, batch_img_metas, rescale=rescale)

        # connvert to DetDataSample
        for i in range(len(results_list)):
            result = DetDataSample()
            result.pred_instances = results_list[i]
            results_list[i] = result
        return results_list

    # TODO: Currently not supported
    def aug_test(self, aug_batch_imgs, aug_batch_img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            aug_batch_imgs (list[Tensor]): The list indicate the
                different augmentation. each item has shape
                of (B, C, H, W).
                Typically these should be mean centered and std scaled.
            aug_batch_img_metas (list[list[dict]]): The outer list
                indicate the test-time augmentations. The inter list indicate
                the batch dimensions.  Each item contains
                the meta information of image with corresponding
                augmentation.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
            The outer list corresponds to each image. The inner list
            corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(aug_batch_imgs)
        results_list = self.bbox_head.aug_test(
            feats, aug_batch_img_metas, rescale=rescale)
        bbox_results = []
        for results in results_list:
            det_bboxes = torch.cat([results.bboxes, results.scores[:, None]],
                                   dim=-1)
            det_labels = results.labels
            bbox_results.append(
                bbox2result(det_bboxes, det_labels,
                            self.bbox_head.num_classes))
        return bbox_results

    # TODO: Currently not supported
    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
