import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule
from torch.nn.modules.utils import _pair as to_2tuple


# Modified from Pytorch-Image-Models
class PatchEmbed(BaseModule):
    """Image to Patch Embedding V2.

    We use a conv layer to implement PatchEmbed.
    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (dict, optional): The config dict for conv layers type
            selection. Default: None.
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Default to be equal with kernel_size).
        padding (int): The padding length of embedding conv. Default: 0.
        dilation (int): The dilation rate of embedding conv. Default: 1.
        pad_to_patch_size (bool, optional): Whether to pad feature map shape
            to multiple patch size. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        self.init_cfg = init_cfg

        if stride is None:
            stride = kernel_size

        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size

        # Use conv layer to embed
        conv_type = conv_type or 'Conv2d'
        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))

        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x
