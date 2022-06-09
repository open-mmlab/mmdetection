# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

from torch import Tensor

from mmdet.core.utils import (ConfigType, OptConfigType, OptMultiConfig,
                              SampleList)
from mmdet.registry import MODELS
from .base import BaseDetector

INF = 1e8


@MODELS.register_module()
class SingleStageInstanceSegmentor(BaseDetector):
    """Base class for single-stage instance segmentors."""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 preprocess_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None
        if bbox_head is not None:
            bbox_head.update(train_cfg=copy.deepcopy(train_cfg))
            bbox_head.update(test_cfg=copy.deepcopy(test_cfg))
            self.bbox_head = MODELS.build(bbox_head)
        else:
            self.bbox_head = None

        assert mask_head, f'`mask_head` must ' \
                          f'be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = MODELS.build(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_bbox_head(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have different
            resolutions.
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
        # bbox_head
        if self.with_bbox_head:
            # TODO: current not supported
            pass
        # mask_head
        mask_outs = self.mask_head(x)
        outs = outs + (mask_outs, )
        return outs

    def forward_train(self, batch_inputs: Tensor,
                      batch_data_samples: SampleList, **kwargs) -> dict:
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
        # process batch_data_samples to set `batch_input_shape` into metainfo
        super().forward_train(
            batch_inputs=batch_inputs, batch_data_samples=batch_data_samples)

        x = self.extract_feat(batch_inputs)
        losses = dict()

        # TODO: Check the logic in CondInst and YOLACT
        # CondInst and YOLACT have bbox_head
        if self.with_bbox_head:
            # TODO: Check whether can simply use bbox.forward_train here
            det_losses, positive_infos = self.bbox_head.forward_train(
                x, batch_data_samples, **kwargs)
            # # bbox_head_preds is a tuple
            # bbox_head_preds = self.bbox_head(x)
            # # positive_infos is a list of obj:`InstanceData`
            # # It contains the information about the positive samples
            # # CondInst, YOLACT
            # det_losses, positive_infos = self.bbox_head.loss(
            #     *bbox_head_preds,
            #     gt_bboxes=gt_bboxes,
            #     gt_labels=gt_labels,
            #     gt_masks=gt_masks,
            #     img_metas=img_metas,
            #     gt_bboxes_ignore=gt_bboxes_ignore,
            #     **kwargs)
            losses.update(det_losses)
        else:
            positive_infos = None

        mask_loss = self.mask_head.forward_train(
            x, batch_data_samples, positive_infos=positive_infos, **kwargs)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)
        return losses

    def simple_test(self,
                    batch_inputs: Tensor,
                    batch_img_metas: List[dict],
                    rescale: bool = False,
                    **kwargs) -> SampleList:
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
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = self.extract_feat(batch_inputs)
        if self.with_bbox_head:
            # TODO: currently not supported, check whether can use
            #  `bbox_head.simple_test` to keep the logic same as
            #  single_stage detector
            results_list = self.bbox_head.simple_test(
                x, batch_img_metas, rescale=rescale)
            # outs = self.bbox_head(x)
            # # results_list is list[obj:`InstanceData`]
            # results_list = self.bbox_head.get_results(
            #     *outs, img_metas=img_metas,
            #     cfg=self.test_cfg, rescale=rescale)
        else:
            results_list = None

        results_list = self.mask_head.simple_test(
            x, batch_img_metas, rescale=rescale, results_list=results_list)

        for results in results_list:
            # create dummy bbox results to store the scores
            if 'bboxes' not in results:
                results.bboxes = results.scores.new_zeros(len(results), 4)

        # connvert to DetDataSample
        results_list = self.postprocess_result(results_list)
        return results_list

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
