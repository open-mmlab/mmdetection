import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList


class ConvUpsample(BaseModule):
    """ConvUpsample performs 2x upsampling after Conv.

    There are several `ConvModule` layers. In the first few layers, upsampling
    will be applied after each layer of convolution. The number of upsampling
    must be no more than the number of ConvModule layers.

    Args:
        in_channels (int): Number of channels in the input feature map.
        inner_channels (int): Number of channels produced by the convolution.
        num_layers (int): Number of convolution layers.
        num_upsample (int | optional): Number of upsampling layer. Must be no
            more than num_layers. Upsampling will be applied after the first
            ``num_upsample`` layers of convolution. Default: ``num_layers``.
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict): Config dict for initialization. Default: None.
        kwargs (key word augments): Other augments used in ConvModule.
    """

    def __init__(self,
                 in_channels,
                 inner_channels,
                 num_layers=1,
                 num_upsample=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(ConvUpsample, self).__init__(init_cfg)
        if num_upsample is None:
            num_upsample = num_layers
        assert num_upsample <= num_layers, \
            f'num_upsample({num_upsample})must be no more than ' \
            f'num_layers({num_layers})'
        self.num_layers = num_layers
        self.num_upsample = num_upsample
        self.conv = ModuleList()
        for i in range(num_layers):
            self.conv.append(
                ConvModule(
                    in_channels,
                    inner_channels,
                    3,
                    padding=1,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            in_channels = inner_channels

    def forward(self, x):
        num_upsample = self.num_upsample
        for i in range(self.num_layers):
            x = self.conv[i](x)
            if num_upsample > 0:
                num_upsample -= 1
                x = F.interpolate(
                    x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
