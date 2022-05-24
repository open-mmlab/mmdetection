# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
from mmcv.utils import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor

from mmdet.core import DetDataSample
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
                 neck: Optional[Union[ConfigDict, dict]] = None,
                 rpn_head: Optional[Union[ConfigDict, dict]] = None,
                 roi_head: Optional[Union[ConfigDict, dict]] = None,
                 train_cfg: Optional[Union[ConfigDict, dict]] = None,
                 test_cfg: Optional[Union[ConfigDict, dict]] = None,
                 pretrained: Optional[str] = None,
                 preprocess_cfg: Optional[Union[ConfigDict, dict]] = None,
                 init_cfg: Optional[Union[ConfigDict, dict]] = None) -> None:
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
        outs = ()
        # backbone
        x = self.extract_feat(batch_inputs)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4, device=batch_inputs.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      batch_inputs: Tensor,
                      batch_data_samples: List[DetDataSample],
                      proposals: Optional[List[InstanceData]] = None,
                      **kwargs) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            proposals (List[:obj:`InstanceData`]): The proposals
                obtained in advance outside. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        super().forward_train(
            batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)
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

            rpn_losses, rpn_results_list = self.rpn_head.forward_train(
                x, rpn_data_samples, proposal_cfg=proposal_cfg, **kwargs)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in keys:
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            # TODO: Need check with Fast R-CNN
            assert proposals is not None
            assert len(proposals) == len(batch_data_samples)
            rpn_results_list = []
            for i in range(len(batch_data_samples)):
                results = InstanceData()
                results.bboxes = proposals[i]
                rpn_results_list.append(results)

        roi_losses = self.roi_head.forward_train(x, rpn_results_list,
                                                 batch_data_samples, **kwargs)
        losses.update(roi_losses)

        return losses

    # TODO: Currently not supported
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
        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self,
                    batch_inputs: Tensor,
                    batch_img_metas: List[dict],
                    proposals: Optional[List[InstanceData]] = None,
                    rescale: bool = False) -> List[DetDataSample]:
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
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)
        if proposals is None:
            rpn_results_list = self.rpn_head.simple_test(
                x, batch_img_metas, rescale=False)
        else:
            rpn_results_list = proposals
            # TODO: Need check with Fast R-CNN
            assert len(rpn_results_list) == len(batch_img_metas)
            for i in range(len(batch_img_metas)):
                results = InstanceData()
                results.bboxes = proposals[i]
                rpn_results_list.append(results)

        results_list = self.roi_head.simple_test(
            x, rpn_results_list, batch_img_metas, rescale=rescale)

        # connvert to DetDataSample
        results_list = self.postprocess_result(results_list)

        return results_list

    # TODO: Currently not supported
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


# TODO: remove this after finish refactor TwoStagePanopticSegmentor
def results2proposal(results_list):
    if isinstance(results_list[0], InstanceData):
        proposal_list = []
        for results in results_list:
            proposal_list.append(
                torch.cat([results.bboxes, results.scores[:, None]], dim=-1))
        return proposal_list
    else:
        return results_list
