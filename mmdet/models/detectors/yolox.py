# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.model import stack_batch

from mmdet.core.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.registry import MODELS
from ...utils import log_img_scale
from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_

    Note: Considering the trade-off between training speed and accuracy,
    multi-scale training is temporarily kept. More elegant implementation
    will be adopted in the future.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLOX. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLOX. Defaults to None.
        input_size (tuple): The model default input image size. The shape
            order should be (height, width). Defaults to (640, 640).
        size_multiplier (int): Image size multiplication factor.
            Defaults to 32.
        random_size_range (tuple): The multi-scale random range during
            multi-scale training. The real training image size will
            be multiplied by size_multiplier. Defaults to (15, 25).
        random_size_interval (int): The iter interval of change
            image size. Defaults to 10.
        preprocess_cfg (:obj:`ConfigDict` or dict, optional): Model
            preprocessing config for processing the input data. it usually
            includes ``to_rgb``, ``pad_size_divisor``, ``pad_value``,
            ``mean`` and ``std``. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 input_size: Tuple[int, int] = (640, 640),
                 size_multiplier: int = 32,
                 random_size_range: Tuple[int, int] = (15, 25),
                 random_size_interval: int = 10,
                 preprocess_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            preprocess_cfg=preprocess_cfg,
            init_cfg=init_cfg)
        log_img_scale(input_size, skip_square=True)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self.pad_size_divisor = size_multiplier
        self._progress_in_iter = 0

    def preprocess_data(self, data: List[dict]) -> tuple:
        """ Process input data during training and simple testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.
                 - batch_inputs (Tensor): The batch input tensor.
                 - batch_data_samples (list[:obj:`DetDataSample`]): The Data
                     Samples. It usually includes information such as
                     `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        """
        inputs = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]
        batch_data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        inputs = [_input.to(self.device) for _input in inputs]
        # YOLOX does not need preprocess_cfg
        batch_inputs = stack_batch(inputs, self._size_multiplier).float()

        # TODO: align with the model design later
        if self.training:
            # resize a batch of images and bboxes to shape ``self._input_size``
            scale_y = self._input_size[0] / self._default_input_size[0]
            scale_x = self._input_size[1] / self._default_input_size[1]
            if scale_x != 1 or scale_y != 1:
                batch_inputs = F.interpolate(
                    batch_inputs,
                    size=self._input_size,
                    mode='bilinear',
                    align_corners=False)
                for data_sample in batch_data_samples:
                    data_sample.set_metainfo({'img_shape': self._input_size})
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
            # get random size every ``self._random_size_interval`` iterations
            # for multi-sale training
            # TODO: use messagehub after mmengine fixes the bug
            if (self._progress_in_iter + 1) % self._random_size_interval == 0:
                self._input_size = self._random_resize()
            self._progress_in_iter += 1
        return batch_inputs, batch_data_samples

    def _random_resize(self) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(self.device)

        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(
                self._default_input_size[1]) / self._default_input_size[0]
            size = (self._size_multiplier * size,
                    self._size_multiplier * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
