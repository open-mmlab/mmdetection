# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.mask import BitmapMasks


@MODELS.register_module()
class TrackDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for tracking tasks.

    Accepts the data sampled by the dataloader, and preprocesses it into the
    format of the model input. ``TrackDataPreprocessor`` provides the
    tracking data pre-processing as follows:

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to inputs.
    - Convert inputs from bgr to rgb if the shape of input is (1, 3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.
    - Record the information of ``batch_input_shape`` and ``pad_shape``.

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
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__()
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                'The length of mean should be 1 or 3 to be compatible with '
                f'RGB or gray image, but got {len(mean)}')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                'The length of std should be 1 or 3 to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)}')

            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(1, -1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(1, -1, 1, 1), False)
        else:
            self._enable_normalize = False

        self.channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> Dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``TrackDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Dict[str, List[torch.Tensor]], OptSampleList]: Data in the
            same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        imgs, data_samples = data['inputs'], data['data_samples']

        # TODO: whether normalize should be after stack_batch
        # The shape of imgs[0] is (T, C, H, W).
        channel = imgs[0].size(1)
        if self.channel_conversion and channel == 3:
            imgs = [_img[:, [2, 1, 0], ...] for _img in imgs]
        # change to `float`
        imgs = [_img.float() for _img in imgs]
        if self._enable_normalize:
            imgs = [(_img - self.mean) / self.std for _img in imgs]

        inputs = stack_batch(imgs, self.pad_size_divisor, self.pad_value)

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs.size()[-2:])
            for track_data_sample, pad_shapes in zip(data_samples,
                                                     batch_pad_shape):
                for i in range(len(track_data_sample)):
                    det_data_sample = track_data_sample[i]
                    det_data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shapes[i]
                    })
            if self.pad_mask:
                self.pad_gt_masks(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                # we only support T==1 when using batch augments.
                # Only yolox need batch_aug, and yolox can only process
                # (N, C, H, W) shape.
                # The shape of `inputs` is (N, T, C, H, W), hence, we use
                # inputs[:, 0] to change the shape to (N, C, H, W).
                assert inputs.size(1) == 1 and len(
                    data_samples[0]
                ) == 1, 'Only support the number of sequence images equals to 1 when using batch augment.'  # noqa: E501
                det_data_samples = [
                    track_data_sample[0] for track_data_sample in data_samples
                ]
                aug_inputs, aug_det_samples = batch_aug(
                    inputs[:, 0], det_data_samples)
                inputs = aug_inputs.unsqueeze(1)
                for track_data_sample, det_sample in zip(
                        data_samples, aug_det_samples):
                    track_data_sample.video_data_samples = [det_sample]

        return dict(inputs=inputs, data_samples=data_samples)

    def _get_pad_shape(self, data: dict) -> Dict[str, List]:
        """Get the pad_shape of each image based on data and pad_size_divisor.

        Args:
            data (dict): Data sampled from dataloader.

        Returns:
            Dict[str, List]: The shape of padding.
        """
        batch_pad_shape = dict()
        batch_pad_shape = []
        for imgs in data['inputs']:
            # The sequence images in one sample among a batch have the same
            # original shape
            pad_h = int(np.ceil(imgs.shape[-2] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(imgs.shape[-1] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_shapes = [(pad_h, pad_w)] * imgs.size(0)
            batch_pad_shape.append(pad_shapes)
        return batch_pad_shape

    def pad_gt_masks(self, data_samples: Sequence[TrackDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in data_samples[0][0].get('gt_instances', None):
            for track_data_sample in data_samples:
                for i in range(len(track_data_sample)):
                    det_data_sample = track_data_sample[i]
                    masks = det_data_sample.gt_instances.masks
                    # TODO: whether to use BitmapMasks
                    assert isinstance(masks, BitmapMasks)
                    batch_input_shape = det_data_sample.batch_input_shape
                    det_data_sample.gt_instances.masks = masks.pad(
                        batch_input_shape, pad_val=self.mask_pad_value)


def stack_batch(tensors: List[torch.Tensor],
                pad_size_divisor: int = 0,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the common height and width
    is divisible by ``pad_size_divisor``.

    Args:
        tensors (List[Tensor]): The input multiple tensors. each is a
            TCHW 4D-tensor. T denotes the number of key/reference frames.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the common height and width is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need a divisibility of 32. Defaults to 0
        pad_value (int, float): The padding value. Defaults to 0

    Returns:
       Tensor: The NTCHW 5D-tensor. N denotes the batch size.
    """
    assert isinstance(tensors, list), \
        f'Expected input type to be list, but got {type(tensors)}'
    assert len(set([tensor.ndim for tensor in tensors])) == 1, \
        f'Expected the dimensions of all tensors must be the same, ' \
        f'but got {[tensor.ndim for tensor in tensors]}'
    assert tensors[0].ndim == 4, f'Expected tensor dimension to be 4, ' \
                                 f'but got {tensors[0].ndim}'
    assert len(set([tensor.shape[0] for tensor in tensors])) == 1, \
        f'Expected the channels of all tensors must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in tensors]}'

    tensor_sizes = [(tensor.shape[-2], tensor.shape[-1]) for tensor in tensors]
    max_size = np.stack(tensor_sizes).max(0)

    if pad_size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (
            max_size +
            (pad_size_divisor - 1)) // pad_size_divisor * pad_size_divisor

    padded_samples = []
    for tensor in tensors:
        padding_size = [
            0, max_size[-1] - tensor.shape[-1], 0,
            max_size[-2] - tensor.shape[-2]
        ]
        if sum(padding_size) == 0:
            padded_samples.append(tensor)
        else:
            padded_samples.append(F.pad(tensor, padding_size, value=pad_value))

    return torch.stack(padded_samples, dim=0)
