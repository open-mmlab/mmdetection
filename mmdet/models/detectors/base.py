# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16
from mmengine.config import ConfigDict
from mmengine.data import InstanceData
from torch import Tensor, device
from torch.optim import Optimizer

from mmdet.core import DetDataSample
from mmdet.core.utils import stack_batch


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
        preprocess_cfg (dict or ConfigDict, optional):
            Model preprocessing config for processing the input data.
            it usually includes ``to_rgb``, ``pad_size_divisor``,
            ``pad_value``, ``mean`` and ``std``. Defaults to None.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 preprocess_cfg: Optional[Union[dict, ConfigDict]] = None,
                 init_cfg: Optional[Union[dict, ConfigDict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = False
        self.preprocess_cfg = preprocess_cfg

        self.pad_size_divisor = 0
        self.pad_value = 0

        if self.preprocess_cfg is not None:
            assert isinstance(self.preprocess_cfg, dict)
            self.preprocess_cfg = copy.deepcopy(self.preprocess_cfg)

            self.to_rgb = preprocess_cfg.get('to_rgb', False)
            self.pad_size_divisor = preprocess_cfg.get('pad_size_divisor', 0)
            self.pad_value = preprocess_cfg.get('pad_value', 0)
            self.register_buffer(
                'pixel_mean',
                torch.tensor(preprocess_cfg['mean']).view(-1, 1, 1), False)
            self.register_buffer(
                'pixel_std',
                torch.tensor(preprocess_cfg['std']).view(-1, 1, 1), False)
        else:
            # Only used to provide device information
            warnings.warn('We treat `model.preprocess_cfg` is None.')
            self.register_buffer('pixel_mean', torch.tensor(1), False)

    @property
    def device(self) -> device:
        """Get the current device."""
        return self.pixel_mean.device

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def extract_feats(self, multi_batch_inputs: List[Tensor]) -> List[Tensor]:
        """Extract features from multiple images.

        Args:
            multi_batch_inputs (list[Tensor]): A list of images.
                The images are augmented from the same image but
                in different ways.

        Returns:
            list[Tensor]: Features of different images.
        """
        assert isinstance(multi_batch_inputs, list)
        return [
            self.extract_feat(batch_inputs)
            for batch_inputs in multi_batch_inputs
        ]

    @auto_fp16(apply_to=('batch_inputs', ))
    def forward_train(self, batch_inputs: Tensor,
                      batch_data_samples: List[DetDataSample],
                      **kwargs) -> None:
        """
        Args:
            batch_inputs (Tensor):The image Tensor should have a shape NxCxHxW.
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(batch_inputs[0].size()[-2:])
        for data_samples in batch_data_samples:
            data_samples.set_metainfo({'batch_input_shape': batch_input_shape})

    async def async_simple_test(self, batch_inputs, batch_img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, batch_inputs: Tensor, batch_img_metas: List[dict],
                    **kwargs):
        pass

    # TODO: Currently not supported
    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    # TODO: Currently not supported
    async def aforward_test(self, *, aug_batch_imgs, aug_batch_data_samples,
                            **kwargs):
        """
        Args:
            aug_batch_imgs (List[Tensor]): the outer list indicates test-time
                augmentations, the Tensor should have a shape NxCxHxW.
                We only support batch size = 1 when do the augtest.
            aug_batch_data_samples (List[List[:obj:`GeneralData`]]): the
                outer list indicates test-time augmentations and inner list
                indicates batch dimension. We only support batch size = 1 when
                do the augtest.
        """
        num_augs = len(aug_batch_data_samples)
        batch_size = len(aug_batch_data_samples[0])

        aug_batch_img_metas = []
        for aug_index in range(num_augs):
            batch_img_metas = []
            for batch_index in range(batch_size):
                single_data_sample = aug_batch_data_samples[aug_index][
                    batch_index]
                batch_img_metas.append(single_data_sample.meta)

            aug_batch_img_metas.append(batch_img_metas)

        for var, name in [(aug_batch_imgs, 'imgs'),
                          (aug_batch_img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(aug_batch_imgs)
        if num_augs != len(aug_batch_img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(aug_batch_imgs), len(aug_batch_img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = aug_batch_imgs[0].size(0)
        assert imgs_per_gpu == 1

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is used for the constructing mask in
        # transformer layer
        for batch_img, batch_img_metas in zip(aug_batch_imgs,
                                              aug_batch_img_metas):
            batch_size = len(batch_img_metas)
            for img_id in range(batch_size):
                batch_img_metas[img_id]['batch_input_shape'] = \
                    tuple(batch_img.size()[-2:])

        if num_augs == 1:
            return await self.async_simple_test(aug_batch_imgs[0],
                                                aug_batch_img_metas[0],
                                                **kwargs)
        else:
            raise NotImplementedError

    def forward(self,
                data: List[dict],
                optimizer: Optional[Union[Optimizer, dict]] = None,
                return_loss: bool = False,
                **kwargs) -> Union[dict, List[DetDataSample]]:
        """The iteration step during training and testing. This method defines
        an iteration step during training and testing, except for the back
        propagation and optimizer updating during training, which are done in
        an optimizer hook.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer`, dict, Optional): The
                optimizer of runner. This argument is unused and reserved.
                Default to None.
            return_loss (bool): Whether to return loss. In general,
                it will be set to True during training and False
                during testing. Default to False.

        Returns:
            during training
                dict: It should contain at least 3 keys: ``loss``,
                ``log_vars``, ``num_samples``.
                    - ``loss`` is a tensor for back propagation, which can be a
                      weighted sum of multiple losses.
                    - ``log_vars`` contains all the variables to be sent to the
                        logger.
                    - ``num_samples`` indicates the batch size (when the model
                        is DDP, it means the batch size on each GPU), which is
                        used for averaging the logs.

            during testing
                list(obj:`DetDataSample`): Detection results of the
                input images. Each DetDataSample usually contains
                ``pred_instances`` or ``pred_panoptic_seg`` or
                ``pred_sem_seg``.
        """
        batch_inputs, batch_data_samples = self.preprocess_data(data)

        if return_loss:
            losses = self.forward_train(batch_inputs, batch_data_samples,
                                        **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(batch_data_samples))
            return outputs
        else:
            # TODO: refactor and support aug test later
            assert isinstance(data[0]['inputs'], torch.Tensor), \
                'Only support simple test currently. Aug-test is ' \
                'not supported yet'
            return self.forward_simple_test(batch_inputs, batch_data_samples,
                                            **kwargs)

    def preprocess_data(self, data: List[dict]) -> tuple:
        """Process input data during training and simple testing phases.

        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple: It should contain 2 item.

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

        if self.preprocess_cfg is None:
            # YOLOX does not need preprocess_cfg
            return stack_batch(inputs).float(), batch_data_samples

        if self.to_rgb and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]
        inputs = [(_input - self.pixel_mean) / self.pixel_std
                  for _input in inputs]
        batch_inputs = stack_batch(inputs, self.pad_size_divisor,
                                   self.pad_value)
        return batch_inputs, batch_data_samples

    def postprocess_result(self, results_list: List[InstanceData]) \
            -> List[DetDataSample]:
        """ Convert results list to `DetDataSample`.
        Args:
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

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
            """
        for i in range(len(results_list)):
            result = DetDataSample()
            result.pred_instances = results_list[i]
            results_list[i] = result
        return results_list

    def preprocess_aug_testing_data(self, data: List[dict]) -> tuple:
        """ Process input data during training and testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader. The list indicate the batch dimension.
                Each dict contains these keys:

                - `img` (list[Tensor]): Image tensor with different test-time
                  augmentation.
                - `data_sample` (list[:obj:`GeneralData`]): Meta information
                  and annotations under different test-time augmentation.


        Returns:
            tuple:  It should contain 2 items.

                 - aug_batch_imgs (list[Tensor]):  List of batch image
                   tensor. The list indicate the test-time augmentations.
                   Note that the batch size always is 1
                   when do the augtest.
                 - aug_batch_data_samples
                   (list[list[:obj:`GeneralData`]], Optional):
                   The Data Samples. It usually includes information such as
                   `gt_instance`. Return None If the input datas does not
                   contain `data_sample`. The outer list indicate the
                   number of augmentations and inter list indicate the
                   batch dimension.
        """

        num_augs = len(data[0]['img'])
        batch_size = len(data)
        aug_batch_imgs = []
        aug_batch_data_samples = []

        # adjust `images` and `data_samples` to a list of list
        # outer list is test-time augmentation and inter list
        # is batch dimension
        for aug_index in range(num_augs):
            batch_imgs = []
            batch_data_samples = []
            for batch_index in range(batch_size):
                single_img = data[batch_index]['img'][aug_index]

                # to gpu and normalize
                single_img = single_img.to(self.device)
                if self.preprocess_cfg is None:
                    # YOLOX does not need preprocess_cfg
                    single_img = single_img.float()
                else:
                    if self.to_rgb and single_img[0].size(0) == 3:
                        single_img = single_img[[2, 1, 0], ...]
                    single_img = (single_img -
                                  self.pixel_mean) / self.pixel_std

                batch_imgs.append(single_img)
                batch_data_samples.append(
                    data[batch_index]['data_sample'][aug_index])
            aug_batch_imgs.append(
                stack_batch(batch_imgs, self.pad_size_divisor, self.pad_value))
            aug_batch_data_samples.append(batch_data_samples)

        return aug_batch_imgs, aug_batch_data_samples

    @auto_fp16(apply_to=('batch_inputs', ))
    def forward_simple_test(self, batch_inputs: Tensor,
                            batch_data_samples: List[DetDataSample],
                            **kwargs) -> List[DetDataSample]:
        """Test function without test time augmentation.

        Args:
            batch_inputs (Tensor): The input Tensor should have a
                shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            list(obj:`DetDataSample`): Detection results of the
            input images. Each DetDataSample usually contains
            ``pred_instances`` or ``pred_panoptic_seg`` or
            ``pred_sem_seg``.
        """
        # TODO: Consider merging with forward_train logic
        batch_size = len(batch_data_samples)
        batch_img_metas = []
        for batch_index in range(batch_size):
            metainfo = batch_data_samples[batch_index].metainfo
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            metainfo['batch_input_shape'] = \
                tuple(batch_inputs.size()[-2:])
            batch_img_metas.append(metainfo)

        return self.simple_test(
            batch_inputs, batch_img_metas, rescale=True, **kwargs)

    def _parse_losses(self, losses: dict) -> tuple:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
