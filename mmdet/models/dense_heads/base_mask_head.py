# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

from mmcv.runner import BaseModule
from torch import Tensor

from mmdet.core.utils import (InstanceList, OptInstanceList, OptMultiConfig,
                              OptSamplingResultList, SampleList)


class BaseMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for mask heads used in One-Stage Instance Segmentation."""

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_results(self, **kwargs):
        """Transform network outputs of a batch into mask results."""
        pass

    def forward_train(self,
                      x: Union[List[Tensor], Tuple[Tensor]],
                      batch_data_samples: SampleList,
                      positive_infos: OptSamplingResultList = None,
                      **kwargs) -> dict:
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            positive_infos (list[:obj:``], optional): Information
                of positive samples. Used when the label assignment is
                done outside the MaskHead, e.g., in BboxHead in
                YOLACT or CondInst, etc. When the label assignment is done in
                MaskHead, it would be None, like SOLO. All values
                in it should have shape (num_positive_samples, *).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if positive_infos is None:
            outs = self(x)
        else:
            # TODO: Currently not checked
            outs = self(x, positive_infos)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'

        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            # pad the `gt_mask` to keep the same shape as `batch_img_shape`
            img_shape = data_sample.metainfo['batch_input_shape']
            gt_masks = data_sample.gt_instances.masks.pad(img_shape)
            data_sample.gt_instances.masks = gt_masks
            batch_gt_instances.append(data_sample.gt_instances)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)

        loss = self.loss(
            *outs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            positive_infos=positive_infos,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            **kwargs)
        return loss

    def simple_test(self,
                    x: Tuple[Tensor],
                    batch_img_metas: List[dict],
                    rescale: bool = False,
                    results_list: OptInstanceList = None,
                    **kwargs) -> InstanceList:
        """Test function without test-time augmentation.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            results_list (list[obj:``], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        if results_list is None:
            outs = self(x)
        else:
            outs = self(x, results_list=results_list)
        results_list = self.get_results(
            *outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            results_list=results_list,
            **kwargs)
        return results_list
