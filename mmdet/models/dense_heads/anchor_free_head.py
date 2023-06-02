# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, List, Sequence, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from numpy import ndarray
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptInstanceList)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..utils import multi_apply
from .base_dense_head import BaseDenseHead

StrideType = Union[Sequence[int], Sequence[Tuple[int, int]]]


@MODELS.register_module()
class AnchorFreeHead(BaseDenseHead):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.).

    Args:
        num_classes (int): 不包括背景类别的类别数.
        in_channels (int): 输入特征图中的通道数.
        feat_channels (int): 隐藏通道数. 用于子类.
        stacked_convs (int): head部分重复的conv数量.
        strides (tuple): 各个特征图的下采样倍数.
        dcn_on_last_conv (bool): If true, 在最后一层使用dcn. Default: False.
        conv_bias (bool | str): 如果值为 `auto`,将由 norm_cfg 决定. 如果 `norm_cfg` 为 None,
            则 conv 的偏差将设置为 True, 否则False. Default: "auto".
        loss_cls (dict): 分类损失的配置.
        loss_bbox (dict): 回归损失的配置.
        bbox_coder (dict): box的编解码配置. Default 'DistancePointBBoxCoder'.
        conv_cfg (dict): 卷积层的配置字典. Default: None.
        norm_cfg (dict): norm层的配置字典. Default: None.
        train_cfg (dict): head的训练配置.
        test_cfg (dict): head的测试配置.
        init_cfg (dict or list[dict], optional): 参数初始化配置字典.
    """  # noqa: W605

    _version = 1

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: StrideType = (4, 8, 16, 32, 64),
        dcn_on_last_conv: bool = False,
        conv_bias: Union[bool, str] = 'auto',
        loss_cls: ConfigType = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
        bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: MultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)

        self.prior_generator = MlvlPointGenerator(strides)

        # 为了保持更通用的接口,和anchor_head保持一致. 我们可以把点想成单个anchor
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """初始化各层结构."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        """初始化分类卷积层."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """初始化回归卷积层."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """初始化预测层."""
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                if len(key) < 2:
                    conv_name = None
                elif key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    conv_name = None
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each scale \
            level, each is a 4D-tensor, the channel number is num_points * 4.
        """
        return multi_apply(self.forward_single, x)[:2]

    def forward_single(self, x):
        """单层级上的前向传播.

        Args:
            x (Tensor): 指定stride的 FPN 特征图.

        Returns:
            tuple: 最终预测的cls, reg,以及分类卷积后的分类特征图,回归卷积后的回归特征图,
                一些模型需要这些返回值,如 FCOS.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    @abstractmethod
    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        """

        raise NotImplementedError

    @abstractmethod
    def get_targets(self, points: List[Tensor],
                    batch_gt_instances: InstanceList) -> Any:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
        """
        raise NotImplementedError

    # TODO refactor aug_test
    def aug_test(self,
                 aug_batch_feats: List[Tensor],
                 aug_batch_img_metas: List[List[Tensor]],
                 rescale: bool = False) -> List[ndarray]:
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            aug_batch_img_metas (list[list[dict]]): the outer list indicates
                test-time augs (multiscale, flip, etc.) and the inner list
                indicates images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(
            aug_batch_feats, aug_batch_img_metas, rescale=rescale)
