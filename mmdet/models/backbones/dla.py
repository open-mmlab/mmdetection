import torch
from torch import nn
from mmcv.cnn import (build_norm_layer, build_conv_layer, build_plugin_layer,
                      constant_init)
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES


class BasicBlock(nn.Module):

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
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm1_name, norm1)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation)

        self.add_module(self.norm2_name, norm2)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, identity=None):

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


class Bottleneck(nn.Module):
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
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.with_dcn = dcn is not None
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                bottle_planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                bottle_planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x, identity=None):

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
        """nn.Module: the normalization layer named "norm1" """
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

        if levels == 1:
            self.root = Root(
                root_dim,
                out_channels,
                root_kernel_size,
                root_residual,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
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

    arch_settings = {
        34: (BasicBlock, [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512]),
        46: (Bottleneck, [1, 1, 1, 2, 2, 1], [16, 32, 64, 64, 128, 256]),
        60: (Bottleneck, [1, 1, 1, 2, 3, 1], [16, 32, 128, 256, 512, 1024]),
        102: (Bottleneck, [1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024]),
        169: (Bottleneck, [1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024])
    }

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

    def make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

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
        super(DLANet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


# def dla34(pretrained=None, **kwargs):  # DLA-34
#     model = DLA(depth=34, **kwargs)
#     return model

# def dla46(pretrained=None, **kwargs):  # DLA-46-C
#     model = DLA(depth=46, **kwargs)
#     return model

# def dla60(pretrained=None, **kwargs):  # DLA-60
#     model = DLA(depth=60, **kwargs)
#     return model

# def dla102(pretrained=None, **kwargs):  # DLA-102
#     model = DLA(depth=102, residual_root=True, **kwargs)
#     return model

# def dla169(pretrained=None, **kwargs):  # DLA-169
#     model = DLA(depth=169, residual_root=True, **kwargs)
#     return model

# import cv2

# dla = dla169(pretrained=None)
# # print(dla)
# dla.load_state_dict(
#     torch.load('/home/amax/Work/Datascript/NetModify/dla169-0914e092.pth'),
#     strict=False)
# a = cv2.imread('/home/amax/Work/Test/mmdetection/demo/demo.jpg')
# a = cv2.resize(a, (224, 224))
# a = torch.from_numpy(a).float()
# a = a.permute((2, 0, 1)).unsqueeze(0)
# b = dla(a)
# print(b[3][0, 0, 0, :])
