# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class ZeroShotClassifier(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,  # num_classes
        zs_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        num_classes = out_features
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(in_features, zs_weight_dim)

        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C
        zs_weight = torch.cat(
            [zs_weight, zs_weight.new_zeros(
                (zs_weight_dim, 1))], dim=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)

        assert self.zs_weight.shape[1] == num_classes + 1, self.zs_weight.shape

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
