# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class BaseSemanticHead(BaseModule, metaclass=ABCMeta):
    """Base module of Semantic Head.

    Args:
        num_classes (int): the number of classes.
        seg_rescale_factor (float): the rescale factor for ``gt_sem_seg``,
            which equals to ``1 / output_strides``. The output_strides is
            for ``seg_preds``. Defaults to  1 / 4.
        init_cfg (Optional[Union[:obj:`ConfigDict`, dict]]): the initialization
            config.
        loss_seg (Union[:obj:`ConfigDict`, dict]): the loss of the semantic
            head.
    """

    def __init__(self,
                 num_classes: int,
                 seg_rescale_factor: float = 1 / 4.,
                 loss_seg: ConfigType = dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_seg = MODELS.build(loss_seg)
        self.num_classes = num_classes
        self.seg_rescale_factor = seg_rescale_factor

    @abstractmethod
    def forward(self, x: Union[Tensor, Tuple[Tensor]]) -> Dict[str, Tensor]:
        """Placeholder of forward function.

        Args:
            x (Tensor): Feature maps.

        Returns:
            Dict[str, Tensor]: A dictionary, including features
                and predicted scores. Required keys: 'seg_preds'
                and 'feats'.
        """
        pass

    @abstractmethod
    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Args:
            x (Tensor): Feature maps.

        Returns:
            Dict[str, Tensor]: The loss of semantic head.
        """
        pass

    def predict(self,
                x: Union[Tensor, Tuple[Tensor]],
                batch_img_metas: List[dict],
                rescale: bool = False) -> List[Tensor]:
        """Test without Augmentation.

        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_img_metas (List[dict]): List of image information.
            rescale (bool): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[Tensor]: semantic segmentation logits.
        """
        seg_preds = self.forward(x)['seg_preds']
        seg_preds = F.interpolate(
            seg_preds,
            size=batch_img_metas[0]['batch_input_shape'],
            mode='bilinear',
            align_corners=False)
        seg_preds = [seg_preds[i] for i in range(len(batch_img_metas))]

        if rescale:
            seg_pred_list = []
            for i in range(len(batch_img_metas)):
                h, w = batch_img_metas[i]['img_shape']
                seg_pred = seg_preds[i][:, :h, :w]

                h, w = batch_img_metas[i]['ori_shape']
                seg_pred = F.interpolate(
                    seg_pred[None],
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False)[0]
                seg_pred_list.append(seg_pred)
        else:
            seg_pred_list = seg_preds

        return seg_pred_list
