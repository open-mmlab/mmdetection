# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor

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
                 backbone: Union[ConfigDict, dict],
                 neck: Optional[Union[ConfigDict, dict]] = None,
                 bbox_head: Optional[Union[ConfigDict, dict]] = None,
                 train_cfg: Optional[Union[ConfigDict, dict]] = None,
                 test_cfg: Optional[Union[ConfigDict, dict]] = None,
                 preprocess_cfg: Optional[Union[ConfigDict, dict]] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Union[ConfigDict, dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
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

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, batch_inputs: Tensor) -> tuple:
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(batch_inputs)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      batch_inputs: Tensor,
                      batch_data_samples: List[DetDataSample],
                      proposals: Optional[InstanceData] = None,
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
        super().forward_train(
            batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.forward_train(x, batch_data_samples, **kwargs)
        return losses

    def simple_test(self,
                    batch_inputs: Tensor,
                    batch_img_metas: List[dict],
                    rescale: bool = False) -> List[DetDataSample]:
        """Test function without test-time augmentation.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.simple_test(
            x, batch_img_metas, rescale=rescale)

        # connvert to DetDataSample
        results_list = self.postprocess_result(results_list)
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

        x = self.extract_feats(aug_batch_imgs)
        results_list = self.bbox_head.aug_test(
            x, aug_batch_img_metas, rescale=rescale)
        bbox_results = []
        for results in results_list:
            det_bboxes = torch.cat([results.bboxes, results.scores[:, None]],
                                   dim=-1)
            det_labels = results.labels
            bbox_results.append(
                bbox2result(det_bboxes, det_labels,
                            self.bbox_head.num_classes))
        return bbox_results
