# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Tuple

import torch
from mmcv.utils import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from .base import BaseDetector


@MODELS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 preprocess_cfg: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            tuple[Tensor]: Multi-level features that may have
                different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img: Tensor) -> Tuple[List[Tensor]]:
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self, img, data_samples, proposals=None, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            data_samples (list[:obj:`GeneralData`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposals (List[Tensor]): The proposals obtained in advance
                outside.  Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, results_list = self.rpn_head.forward_train(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # TODO: losses check whether get 'rpn_'
            losses.update(rpn_losses)
            # TODO: remove this after refactor two stage input
            proposal_list = results2proposal(results_list)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, proposal_list,
                                                 data_samples, **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals
        # TODO: remove this after refactor two stage input
        proposal_list = results2proposal(proposal_list)
        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            results_list = self.rpn_head.simple_test(x, img_metas)
            proposal_list = results2proposal(results_list)
        else:
            proposal_list = proposals
        # TODO: remove this after refactor two stage input

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, aug_batch_imgs, aug_batch_img_metas, rescale=False):
        """Test with augmentations.

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

        Returns:
            list(obj:`InstanceData`): Detection results of the
            input images. Each item usually contains\
            following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance,)
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances,).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).


        Note:
            If rescale is False, then returned bboxes and masks will fit
            the scale of aug_imgs[0].
        """
        x = self.extract_feats(aug_batch_imgs)
        # results_list will be in original scale
        results_list = self.rpn_head.aug_test(
            x, aug_batch_img_metas, rescale=True, with_ori_nms=True)

        # TODO: remove this after refactor two stage input
        proposal_list = results2proposal(results_list)

        # TODO support batch for two stage
        assert len(proposal_list) == 1
        return self.roi_head.aug_test(
            x, proposal_list, aug_batch_img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )


# TODO: remove this after refactor `roi_head` input
def results2proposal(results_list):
    if isinstance(results_list[0], InstanceData):
        proposal_list = []
        for results in results_list:
            proposal_list.append(
                torch.cat([results.bboxes, results.scores[:, None]], dim=-1))
        return proposal_list
    else:
        return results_list
