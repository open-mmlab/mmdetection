# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmdet.registry import MODELS


@MODELS.register_module()
class DetDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
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
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

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
        inputs, batch_data_samples = self.collate_data(data)
        batch_pad_shape = self._get_pad_shape(inputs)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]
        else:
            inputs = [_input.float() for _input in inputs]

        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)

        if training and self.batch_augments is not None:
            inputs, batch_data_samples = self.batch_augments(
                inputs, batch_data_samples)

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

        return batch_inputs, batch_data_samples

    def _get_pad_shape(self, ori_inputs: List[torch.Tensor]) -> List[tuple]:
        """Get the pad_shape of each image based on ori_inputs and
        pad_size_divisor."""
        batch_pad_shape = []
        for ori_input in ori_inputs:
            pad_h = int(np.ceil(ori_input.shape[1] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(np.ceil(ori_input.shape[2] /
                                self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape.append((pad_h, pad_w))
        return batch_pad_shape
