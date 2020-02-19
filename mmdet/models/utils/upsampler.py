import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.ops.carafe import CARAFEPack


class PixelShufflePack(nn.Module):

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


upsampler_cfg = {
    # format: layer_type: (abbreviation, module)
    'nearest': ('nearest', nn.Upsample),
    'bilinear': ('bilinear', nn.Upsample),
    'deconv': ('deconv', nn.ConvTranspose2d),
    'pixel_shuffle': ('pixel_shuffle', PixelShufflePack),
    'carafe': ('carafe', CARAFEPack)
}


def build_upsampler_layer(cfg, postfix=''):
    """ Build upsampler layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify upsampler layer type.
            upsample ratio (int): upsample ratio
            layer args: args needed to instantiate a upsampler layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created upsampler layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in upsampler_cfg:
        raise KeyError('Unrecognized upsampler type {}'.format(layer_type))
    else:
        abbr, upsampler = upsampler_cfg[layer_type]
        if upsampler is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    layer = upsampler(**cfg_)
    return name, layer
