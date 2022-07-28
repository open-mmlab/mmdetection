# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.data import PixelData
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import ConfigType


@MODELS.register_module()
class DetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """
        batch_inputs, batch_data_samples = super().forward(
            data=data, training=training)
        batch_pad_shape = self._get_pad_shape(data)

        if batch_data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(batch_inputs[0].size()[-2:])
            for data_samples, pad_shape in zip(batch_data_samples,
                                               batch_pad_shape):
                data_samples.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.pad_mask:
                self.pad_gt_masks(batch_data_samples)

            if self.pad_seg:
                self.pad_gt_sem_seg(batch_data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                batch_inputs, batch_data_samples = batch_aug(
                    batch_inputs, batch_data_samples)

        return batch_inputs, batch_data_samples

    def _get_pad_shape(self, data: Sequence[dict]) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        ori_inputs = [_data['inputs'] for _data in data]
        batch_pad_shape = []
        for ori_input in ori_inputs:
            pad_h = int(np.ceil(ori_input.shape[1] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(ori_input.shape[2] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape.append((pad_h, pad_w))
        return batch_pad_shape

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


@MODELS.register_module()
class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 10,
                 size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor),
                                   round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(
        self, batch_inputs: Tensor, batch_data_samples: List[DetDataSample]
    ) -> Tuple[Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = batch_inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            batch_inputs = F.interpolate(
                batch_inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for data_sample in batch_data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })
                data_sample.gt_instances.bboxes[
                    ...,
                    0::2] = data_sample.gt_instances.bboxes[...,
                                                            0::2] * scale_x
                data_sample.gt_instances.bboxes[
                    ...,
                    1::2] = data_sample.gt_instances.bboxes[...,
                                                            1::2] * scale_y
                if 'ignored_instances' in data_sample:
                    data_sample.ignored_instances.bboxes[
                        ..., 0::2] = data_sample.ignored_instances.bboxes[
                            ..., 0::2] * scale_x
                    data_sample.ignored_instances.bboxes[
                        ..., 1::2] = data_sample.ignored_instances.bboxes[
                            ..., 1::2] * scale_y
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=batch_inputs.device)
        return batch_inputs, batch_data_samples

    def _get_random_size(self, aspect_ratio: float,
                         device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size


@MODELS.register_module()
class BatchFixedSizePad(nn.Module):
    """Fixed size padding for batch images.

    Args:
        size (Tuple[int, int]): Fixed padding size. Expected padding
            shape (h, w). Defaults to None.
        img_pad_value (int): The padded pixel value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
    """

    def __init__(self,
                 size: Tuple[int, int],
                 img_pad_value: int = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255) -> None:
        super().__init__()
        self.size = size
        self.pad_mask = pad_mask
        self.pad_seg = pad_seg
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.seg_pad_value = seg_pad_value

    def forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: Optional[List[dict]] = None
    ) -> Tuple[Tensor, Optional[List[dict]]]:
        """Pad image, instance masks, segmantic segmentation maps."""
        src_h, src_w = batch_inputs.shape[-2:]
        dst_h, dst_w = self.size

        if src_h >= dst_h and src_w >= dst_w:
            return batch_inputs, batch_data_samples

        batch_inputs = F.pad(
            batch_inputs,
            pad=(0, max(0, dst_w - src_w), 0, max(0, dst_h - src_h)),
            mode='constant',
            value=self.img_pad_value)

        if batch_data_samples is not None:
            # update batch_input_shape
            for data_samples in batch_data_samples:
                data_samples.set_metainfo({
                    'batch_input_shape': (dst_h, dst_w),
                    'pad_shape': (dst_h, dst_w)
                })

            if self.pad_mask:
                for data_samples in batch_data_samples:
                    masks = data_samples.gt_instances.masks
                    data_samples.gt_instances.masks = masks.pad(
                        (dst_h, dst_w), pad_val=self.mask_pad_value)

            if self.pad_seg:
                for data_samples in batch_data_samples:
                    gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    gt_sem_seg = F.pad(
                        gt_sem_seg,
                        pad=(0, max(0, dst_w - w), 0, max(0, dst_h - h)),
                        mode='constant',
                        value=self.seg_pad_value)
                    data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

        return batch_inputs, batch_data_samples


@MODELS.register_module()
class MultiBranchDataPreprocessor(BaseDataPreprocessor):
    """DataPreprocessor wrapper for multi-branch data.

    Args:
        data_preprocessor (:obj:`ConfigDict` or dict): Config of
            :class:`DetDataPreprocessor` to process the input data.
    """

    def __init__(self, data_preprocessor: ConfigType) -> None:
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(
        self,
        data: Sequence[dict],
        training: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Optional[list]]]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor`` for multi-branch data.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[torch.Tensor], Dict[Optional[list]]]: Each tuple of
            zip(dict, dict) is the data in the same format as the model input.
        """

        if training is False:
            return self.data_preprocessor(data, training)
        multi_branch_data = {}
        for multi_results in data:
            for branch, results in multi_results.items():
                if multi_branch_data.get(branch, None) is None:
                    multi_branch_data[branch] = [results]
                else:
                    multi_branch_data[branch].append(results)
        multi_batch_inputs, multi_batch_data_samples = {}, {}
        for branch, data in multi_branch_data.items():
            multi_batch_inputs[branch], multi_batch_data_samples[
                branch] = self.data_preprocessor(data, training)
        return multi_batch_inputs, multi_batch_data_samples

    @property
    def device(self):
        return self.data_preprocessor.device

    def to(self, device: Optional[Union[int, torch.device]], *args,
           **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Args:
            device (int or torch.device, optional): The desired device of the
                parameters and buffers in this module.

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.to(device, *args, **kwargs)

    def cuda(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs) -> nn.Module:
        """Overrides this method to set the :attr:`device`

        Returns:
            nn.Module: The model itself.
        """

        return self.data_preprocessor.cpu(*args, **kwargs)
