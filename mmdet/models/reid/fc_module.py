# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmdet.registry import MODELS


@MODELS.register_module()
class FcModule(BaseModule):
    """Fully-connected layer module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Ourput channels.
        norm_cfg (dict, optional): Configuration of normlization method
            after fc. Defaults to None.
        act_cfg (dict, optional): Configuration of activation method after fc.
            Defaults to dict(type='ReLU').
        inplace (bool, optional): Whether inplace the activatation module.
            Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to dict(type='Kaiming', layer='Linear').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: dict = None,
                 act_cfg: dict = dict(type='ReLU'),
                 inplace: bool = True,
                 init_cfg=dict(type='Kaiming', layer='Linear')):
        super(FcModule, self).__init__(init_cfg)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        self.fc = nn.Linear(in_channels, out_channels)
        # build normalization layers
        if self.with_norm:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self):
        """Normalization."""
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        """Model forward."""
        x = self.fc(x)
        if norm and self.with_norm:
            x = self.norm(x)
        if activate and self.with_activation:
            x = self.activate(x)
        return x
