# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Tuple, Union

import mmcv
import torch
from mmengine.config import ConfigDict
from torch import Tensor

from mmdet.core import DetDataSample, bbox_mapping
from mmdet.registry import MODELS
from .base import BaseDetector


@MODELS.register_module()
class RPN(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 backbone: Union[ConfigDict, dict],
                 neck: Union[ConfigDict, dict],
                 rpn_head: Union[ConfigDict, dict],
                 train_cfg: Union[ConfigDict, dict],
                 test_cfg: Union[ConfigDict, dict],
                 pretrained: Optional[str] = None,
                 preprocess_cfg: Optional[ConfigDict] = None,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck) if neck is not None else None
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_num_classes = rpn_head.get('num_classes', 1)
        if rpn_head_num_classes != 1:
            warnings.warn('The `num_classes` should be 1 in RPN, but get '
                          f'{rpn_head_num_classes}, please set '
                          'rpn_head.num_classes = 1 in your config file.')
            rpn_head.update(num_classes=1)
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=test_cfg.rpn)
        self.rpn_head = MODELS.build(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, batch_inputs: Tensor) -> Tuple[List[Tensor]]:
        """Dummy forward function."""
        x = self.extract_feat(batch_inputs)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

    def forward_train(self, batch_inputs: Tensor,
                      batch_data_samples: List[DetDataSample],
                      **kwargs) -> dict:
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
        super().forward_train(batch_inputs, batch_data_samples)

        x = self.extract_feat(batch_inputs)
        # set cat_id of gt_labels to 0 in RPN
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        losses = self.rpn_head.forward_train(x, rpn_data_samples, **kwargs)
        return losses

    def simple_test(self,
                    batch_inputs: Tensor,
                    batch_img_metas: List[dict],
                    rescale: bool = False) \
            -> List[DetDataSample]:
        """Test function without test time augmentation.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_img_metas (list[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.rpn_head.simple_test(
            x, batch_img_metas, rescale=rescale)

        # connvert to DetDataSample
        results_list = self.postprocess_result(results_list)

        return results_list

    # TODO: Currently not supported
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        proposal_list = self.rpn_head.aug_test_rpn(
            self.extract_feats(imgs), img_metas)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape,
                                                scale_factor, flip,
                                                flip_direction)
        return [proposal.cpu().numpy() for proposal in proposal_list]

    # TODO: Currently not supported
    def show_result(self, data, result, top_k=20, **kwargs):
        """Show RPN proposals on the image.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            top_k (int): Plot the first k bboxes only
               if set positive. Default: 20

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if kwargs is not None:
            kwargs.pop('score_thr', None)
            kwargs.pop('text_color', None)
            kwargs['colors'] = kwargs.pop('bbox_color', 'green')
        mmcv.imshow_bboxes(data, result, top_k=top_k, **kwargs)
