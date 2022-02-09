# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.utils import stack_batch
from mmdet.core.visualization import imshow_det_bboxes


class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, img_norm_cfg, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.to_rgb = img_norm_cfg['to_rgb']
        self.register_buffer('pixel_mean',
                             torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1),
                             False)
        self.register_buffer('pixel_std',
                             torch.tensor(img_norm_cfg['std']).view(-1, 1, 1),
                             False)

    @property
    def device(self):
        return self.pixel_mean.device

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    @auto_fp16(apply_to=('imgs', ))
    def forward_train(self, imgs, data_samples, **kwargs):
        """
        Args:
            imgs (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for sample in data_samples:
            sample.meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

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

    # TODO:  auto_fp16 supports context embedding
    @auto_fp16(apply_to=('imgs', ))
    def forward_test(self, aug_batch_imgs, aug_batch_data_samples, **kwargs):
        """
        Args:
            aug_batch_imgs (List[Tensor]): the outer list indicates test-time
                augmentations, the Tensor should have a shape NxCxHxW.
                We only support batch size = 1 when do the augtest.
            aug_batch_data_samples (List[List[:obj:`GeneralData`]]): the
                outer list indicates test-time augmentations and inner list
                indicates batch dimension. We only support batch size = 1 when
                do the augtest.

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

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for batch_img, batch_img_metas in zip(aug_batch_imgs,
                                              aug_batch_img_metas):
            batch_size = len(batch_img_metas)
            for img_id in range(batch_size):
                batch_img_metas[img_id]['batch_input_shape'] = \
                    tuple(batch_img.size()[-2:])

        if num_augs == 1:
            return self.simple_test(aug_batch_imgs[0], aug_batch_img_metas[0],
                                    **kwargs)
        else:
            assert 'proposals' not in kwargs, '`self.aug_test` do not ' \
                                              'support pre-difined proposals'
            aug_results = self.aug_test(aug_batch_imgs, aug_batch_img_metas,
                                        **kwargs)
            return aug_results

    def forward(self, data, optimizer=None, return_loss=True, **kwargs):
        """The iteration step during training. This method defines an iteration
        step during training, except for the back propagation and optimizer
        updating, which are done in an optimizer hook. Note that in some
        complicated cases or models, the whole process including back
        propagation and optimizer updating is also defined in this method, such
        as GAN.

        Args:
            data (list[dict]): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        if torch.onnx.is_in_onnx_export():
            images, data_samples = self.preprocss_testing_data(data)
            assert len(images) == 1
            return self.onnx_export(images[0], data_samples[0])

        if return_loss:
            batch_image, data_samples = self.preprocss_training_data(data)
            losses = self.forward_train(batch_image, data_samples, **kwargs)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data_samples))

            return outputs
        else:
            aug_batch_imgs, aug_batch_data_samples = \
                self.preprocss_testing_data(data)
            return self.forward_test(aug_batch_imgs, aug_batch_data_samples,
                                     **kwargs)

    def preprocss_training_data(self, data):
        """ Process input data during training and testing phases.
        Args:
            data (list[dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.
                 - img (Tensor): The batch image tensor.
                 - data_samples (list[:obj:`GeneralData`], Optional): The Data
                   Samples. It usually includes information such as
                   `gt_instance`. Return None If the input datas does not
                   contain `data_sample`.
        """
        images = [data_['img'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        data_samples = [
            data_sample.to(self.device) for data_sample in data_samples
        ]
        images = [img.to(self.device) for img in images]
        if self.to_rgb and images[0].size(0) == 3:
            images = [image[[2, 1, 0], ...] for image in images]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        batch_image = stack_batch(images)

        return batch_image, data_samples

    def preprocss_testing_data(self, data):
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
                if self.to_rgb and single_img[0].size(0) == 3:
                    single_img = single_img[[2, 1, 0], ...]
                single_img = (single_img - self.pixel_mean) / self.pixel_std

                batch_imgs.append(single_img)
                batch_data_samples.append(
                    data[batch_index]['data_sample'][aug_index])
            aug_batch_imgs.append(stack_batch(batch_imgs))
            aug_batch_data_samples.append(batch_data_samples)

        return aug_batch_imgs, aug_batch_data_samples

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
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

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    # TODO
    def onnx_export(self, img, data_sample):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')
