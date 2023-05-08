# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import mmcls
    from mmcls.evaluation.metrics import Accuracy
except ImportError:
    mmcls = None

from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.structures import ReIDDataSample
from .fc_module import FcModule


@MODELS.register_module()
class LinearReIDHead(BaseModule):
    """Linear head for re-identification.

    Args:
        num_fcs (int): Number of fcs.
        in_channels (int): Number of channels in the input.
        fc_channels (int): Number of channels in the fcs.
        out_channels (int): Number of channels in the output.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to None.
        num_classes (int, optional): Number of the identities. Default to None.
        loss_cls (dict, optional): Cross entropy loss to train the ReID module.
            Defaults to None.
        loss_triplet (dict, optional): Triplet loss to train the ReID module.
            Defaults to None.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to dict(type='Normal',layer='Linear', mean=0, std=0.01,
            bias=0).
    """

    def __init__(self,
                 num_fcs: int,
                 in_channels: int,
                 fc_channels: int,
                 out_channels: int,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 num_classes: Optional[int] = None,
                 loss_cls: Optional[dict] = None,
                 loss_triplet: Optional[dict] = None,
                 topk: Union[int, Tuple[int]] = (1, ),
                 init_cfg: Union[dict, List[dict]] = dict(
                     type='Normal', layer='Linear', mean=0, std=0.01, bias=0)):
        if mmcls is None:
            raise RuntimeError('Please run "pip install openmim" and '
                               'run "mim install mmcls>=1.0.0rc0" tp '
                               'install mmcls first.')
        super(LinearReIDHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

        if loss_cls is None:
            if isinstance(num_classes, int):
                warnings.warn('Since cross entropy is not set, '
                              'the num_classes will be ignored.')
            if loss_triplet is None:
                raise ValueError('Please choose at least one loss in '
                                 'triplet loss and cross entropy loss.')
        elif not isinstance(num_classes, int):
            raise TypeError('The num_classes must be a current number, '
                            'if there is cross entropy loss.')
        self.loss_cls = MODELS.build(loss_cls) if loss_cls else None
        self.loss_triplet = MODELS.build(loss_triplet) \
            if loss_triplet else None

        self.num_fcs = num_fcs
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.num_classes = num_classes

        self._init_layers()

    def _init_layers(self):
        """Initialize fc layers."""
        self.fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            in_channels = self.in_channels if i == 0 else self.fc_channels
            self.fcs.append(
                FcModule(in_channels, self.fc_channels, self.norm_cfg,
                         self.act_cfg))
        in_channels = self.in_channels if self.num_fcs == 0 else \
            self.fc_channels
        self.fc_out = nn.Linear(in_channels, self.out_channels)
        if self.loss_cls:
            self.bn = nn.BatchNorm1d(self.out_channels)
            self.classifier = nn.Linear(self.out_channels, self.num_classes)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        # Multiple stage inputs are acceptable
        # but only the last stage will be used.
        feats = feats[-1]

        for m in self.fcs:
            feats = m(feats)
        feats = self.fc_out(feats)
        return feats

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ReIDDataSample]) -> dict:
        """Calculate losses.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
            data_samples (List[ReIDDataSample]): The annotation data of
                every samples.

        Returns:
            dict: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        losses = self.loss_by_feat(feats, data_samples)
        return losses

    def loss_by_feat(self, feats: torch.Tensor,
                     data_samples: List[ReIDDataSample]) -> dict:
        """Unpack data samples and compute loss."""
        losses = dict()
        gt_label = torch.cat([i.gt_label.label for i in data_samples])
        gt_label = gt_label.to(feats.device)

        if self.loss_triplet:
            losses['triplet_loss'] = self.loss_triplet(feats, gt_label)

        if self.loss_cls:
            feats_bn = self.bn(feats)
            cls_score = self.classifier(feats_bn)
            losses['ce_loss'] = self.loss_cls(cls_score, gt_label)
            acc = Accuracy.calculate(cls_score, gt_label, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(
            self,
            feats: Tuple[torch.Tensor],
            data_samples: List[ReIDDataSample] = None) -> List[ReIDDataSample]:
        """Inference without augmentation.

        Args:
            feats (Tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used.
            data_samples (List[ReIDDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[ReIDDataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        feats = self(feats)

        # The part can not be traced by torch.fx
        data_samples = self.predict_by_feat(feats, data_samples)

        return data_samples

    def predict_by_feat(
            self,
            feats: torch.Tensor,
            data_samples: List[ReIDDataSample] = None) -> List[ReIDDataSample]:
        """Add prediction features to data samples."""
        if data_samples is not None:
            for data_sample, feat in zip(data_samples, feats):
                data_sample.pred_feature = feat
        else:
            data_samples = []
            for feat in feats:
                data_sample = ReIDDataSample()
                data_sample.pred_feature = feat
                data_samples.append(data_sample)

        return data_samples
