import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops.carafe import CARAFEPack


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        scale_factor (int): Upsample ratio
        upsample_kernel (int): Kernel size of Conv layer to expand the channels

    Returns:
        upsampled feature map
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


upsample_cfg = {
    # format: layer_type: (abbreviation, module)
    'nearest': nn.Upsample,
    'bilinear': nn.Upsample,
    'deconv': nn.ConvTranspose2d,
    'pixel_shuffle': PixelShufflePack,
    'carafe': CARAFEPack
}


def build_upsample_layer(cfg):
    """ Build upsample layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify upsample layer type.
            upsample ratio (int): Upsample ratio
            layer args: args needed to instantiate a upsample layer.

    Returns:
        layer (nn.Module): Created upsample layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in upsample_cfg:
        raise KeyError('Unrecognized upsample type {}'.format(layer_type))
    else:
        upsample = upsample_cfg[layer_type]
        if upsample is None:
            raise NotImplementedError

    layer = upsample(**cfg_)
    return layer
