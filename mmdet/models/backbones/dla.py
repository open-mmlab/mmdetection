import torch
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, constant_init
from mmcv.runner import load_checkpoint
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock as _BasicBlock
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class BasicBlock(_BasicBlock):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            plugins=plugins)

    def forward(self, x, identity=None):
        """Forward function."""

        def _inner_forward(x, identity):

            if identity is None:
                identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, identity)
        else:
            out = _inner_forward(x, identity)

        out = self.relu(out)

        return out


class Bottleneck(_Bottleneck):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(Bottleneck, self).__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            plugins=plugins)
        """Bottleneck block for DLA."""

        expansion = self.expansion
        bottle_planes = planes // expansion

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, bottle_planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, bottle_planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            bottle_planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                bottle_planes,
                bottle_planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                bottle_planes,
                bottle_planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg, bottle_planes, planes, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        if self.with_plugins:
            for name in self.after_conv1_plugin_names:
                delattr(self, name)
            for name in self.after_conv2_plugin_names:
                delattr(self, name)
            for name in self.after_conv3_plugin_names:
                delattr(self, name)
            self.after_conv1_plugin_names = self.make_block_plugins(
                bottle_planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                bottle_planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes, self.after_conv3_plugins)

    def forward(self, x, identity=None):
        """Forward function."""

        def _inner_forward(x, identity):
            if identity is None:
                identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, identity)
        else:
            out = _inner_forward(x, identity)

        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 residual,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(Root, self).__init__()
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    @property
    def norm(self):
        """nn.Module: the normalization layer named "norm" """
        return getattr(self, self.norm_name)

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 dcn=None,
                 plugins=None,
                 style='pytorch'):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                dcn=dcn,
                plugins=plugins,
                style=style)
            self.tree2 = block(
                out_channels,
                out_channels,
                1,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                dcn=dcn,
                plugins=plugins,
                style=style)
            self.root = Root(
                root_dim,
                out_channels,
                root_kernel_size,
                root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                dcn=dcn,
                plugins=plugins,
                style=style)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                dcn=dcn,
                plugins=plugins,
                style=style)

        self.level_root = level_root
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels and self.levels == 1:
            self.project = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLANet(nn.Module):
    """DLA backbone.

    Args:
        depth (int): Depth of dla, from {34, 46, 60, 102, 169}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): dla stages. Default: 4.
        out_indices (Sequence[int]): Output from which stages.
        strides (Sequence[int]): Strides of the first block of each stage.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        conv_cfg (dict): Dictionary to construct and config convolution layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        dcn (dict): Dictionary to construct and config deformable convolution
            layer
        stage_with_dcn (tuple[bool]): Stages to apply dcn, length
            should be same as 'num_stages'.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        stage_with_level_root (tuple[bool]): Stages to apply level_root, length
            should be same as 'num_stages'.
        residual_root (bool): whether to use residual in root layer

    Example:
        >>> from mmdet.models import DLANet
        >>> import torch
        >>> self = DLANet(depth=60)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 128, 8, 8)
        (1, 256, 4, 4)
        (1, 512, 2, 2)
        (1, 1024, 1, 1)
    """

    arch_settings = {
        34: (BasicBlock, [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512]),
        46: (Bottleneck, [1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256]),
        60: (Bottleneck, [1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024]),
        102: (Bottleneck, [1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024]),
        169: (Bottleneck, [1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024])
    }

    make_stage_plugins = ResNet.make_stage_plugins

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 strides=(2, 2, 2, 2),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 stage_with_level_root=(False, True, True, True),
                 residual_root=False):
        super(DLANet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for DLA')
        block, levels, channels = self.arch_settings[depth]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.base_layer = nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False),
            build_norm_layer(self.norm_cfg, channels[0])[1],
            nn.ReLU(inplace=True))

        for i in range(2):
            level_layer = self._make_conv_level(
                channels[0], channels[i], levels[i], stride=i + 1)
            layer_name = f'level{i}'
            self.add_module(layer_name, level_layer)

        for i in range(self.num_stages):
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            dla_layer = Tree(
                levels[i + 2],
                block,
                channels[i + 1],
                channels[i + 2],
                strides[i],
                level_root=stage_with_level_root[i],
                root_residual=residual_root,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                style=self.style)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, dla_layer)

        self._freeze_stages()

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                build_norm_layer(self.norm_cfg, planes)[1],
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.base_layer.eval()
            for param in self.base_layer.parameters():
                param.requires_grad = False

            for i in range(2):
                m = getattr(self, f'level{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, torch.tensor(2. / n).sqrt())
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.base_layer(x)
        for i in range(2):
            x = getattr(self, 'level{}'.format(i))(x)
        outs = []
        for i, index in enumerate(range(1, self.num_stages + 1)):
            x = getattr(self, 'layer{}'.format(index))(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(DLANet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
