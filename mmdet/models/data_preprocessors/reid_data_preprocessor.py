# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmdet.registry import MODELS

try:
    import mmpretrain
    from mmpretrain.models.utils.batch_augments import RandomBatchAugment
    from mmpretrain.structures import (batch_label_to_onehot, cat_batch_labels,
                                       tensor_split)
except ImportError:
    mmpretrain = None


def stack_batch_scores(elements, device=None):
    """Stack the ``score`` of a batch of :obj:`LabelData` to a tensor.

    Args:
        elements (List[LabelData]): A batch of :obj`LabelData`.
        device (torch.device, optional): The output device of the batch label.
            Defaults to None.
    Returns:
        torch.Tensor: The stacked score tensor.
    """
    item = elements[0]
    if 'score' not in item._data_fields:
        return None

    batch_score = torch.stack([element.score for element in elements])
    if device is not None:
        batch_score = batch_score.to(device)
    return batch_score


@MODELS.register_module()
class ReIDDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for classification tasks.

    Comparing with the :class:`mmengine.model.ImgDataPreprocessor`,

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
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        to_onehot (bool): Whether to generate one-hot format gt-labels and set
            to data samples. Defaults to False.
        num_classes (int, optional): The number of classes. Defaults to None.
        batch_augments (dict, optional): The batch augmentations settings,
            including "augments" and "probs". For more details, see
            :class:`mmpretrain.models.RandomBatchAugment`.
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        if mmpretrain is None:
            raise RuntimeError('Please run "pip install openmim" and '
                               'run "mim install mmpretrain" to '
                               'install mmpretrain first.')
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                'preprocessing, please specify both `mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if batch_augments is not None:
            self.batch_augments = RandomBatchAugment(**batch_augments)
            if not self.to_onehot:
                from mmengine.logging import MMLogger
                MMLogger.get_current_instance().info(
                    'Because batch augmentations are enabled, the data '
                    'preprocessor automatically enables the `to_onehot` '
                    'option to generate one-hot format labels.')
                self.to_onehot = True
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs = self.cast_data(data['inputs'])

        if isinstance(inputs, torch.Tensor):
            # The branch if use `default_collate` as the collate_fn in the
            # dataloader.

            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant',
                               self.pad_value)
        else:
            # The branch if use `pseudo_collate` as the collate_fn in the
            # dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(processed_inputs, self.pad_size_divisor,
                                 self.pad_value)

        data_samples = data.get('data_samples', None)
        sample_item = data_samples[0] if data_samples is not None else None
        if 'gt_label' in sample_item:
            gt_labels = [sample.gt_label for sample in data_samples]
            gt_labels_tensor = [gt_label.label for gt_label in gt_labels]
            batch_label, label_indices = cat_batch_labels(gt_labels_tensor)
            batch_label = batch_label.to(self.device)

            batch_score = stack_batch_scores(gt_labels, device=self.device)
            if batch_score is None and self.to_onehot:
                assert batch_label is not None, \
                    'Cannot generate onehot format labels because no labels.'
                num_classes = self.num_classes or data_samples[0].get(
                    'num_classes')
                assert num_classes is not None, \
                    'Cannot generate one-hot format labels because not set ' \
                    '`num_classes` in `data_preprocessor`.'
                batch_score = batch_label_to_onehot(batch_label, label_indices,
                                                    num_classes)

            # ----- Batch Augmentations ----
            if training and self.batch_augments is not None:
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(
                        data_samples, tensor_split(batch_label,
                                                   label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)

        return {'inputs': inputs, 'data_samples': data_samples}
