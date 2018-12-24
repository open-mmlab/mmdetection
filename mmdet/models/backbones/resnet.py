import logging
import pickle

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from ..utils import build_norm_layer


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='GN')):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)

        norm_layers = []
        norm_layers.append(build_norm_layer(normalize, planes))
        norm_layers.append(build_norm_layer(normalize, planes))
        self.norm_names = (['gn1', 'gn2'] if normalize['type'] == 'GN'
                           else ['bn1', 'bn2'])
        for name, layer in zip(self.norm_names, norm_layers):
            self.add_module(name, layer)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = getattr(self, self.norm_names[0])(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = getattr(self, self.norm_names[1])(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN')):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        norm_layers = []
        norm_layers.append(build_norm_layer(normalize, planes))
        norm_layers.append(build_norm_layer(normalize, planes))
        norm_layers.append(build_norm_layer(normalize, planes*self.expansion))
        self.norm_names = (['gn1', 'gn2', 'gn3'] if normalize['type'] == 'GN'
                           else ['bn1', 'bn2', 'bn3'])
        for name, layer in zip(self.norm_names, norm_layers):
            self.add_module(name, layer)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = getattr(self, self.norm_names[0])(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = getattr(self, self.norm_names[1])(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = getattr(self, self.norm_names[2])(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   normalize=dict(type='BN')):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(normalize, planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            with_cp=with_cp,
            normalize=normalize))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style,
                  with_cp=with_cp, normalize=normalize))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(
                     type='BN',
                     bn_eval=True,
                     bn_frozen=False),
                 with_cp=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages

        assert isinstance(normalize, dict) and 'type' in normalize
        assert normalize['type'] in ['BN', 'GN']
        if normalize['type'] == 'GN':
            assert 'num_groups' in normalize
        else:
            assert (set(['type', 'bn_eval', 'bn_frozen'])
                    == set(normalize))
        if normalize['type'] == 'BN':
            self.bn_eval = normalize['bn_eval']
            self.bn_frozen = normalize['bn_frozen']
        self.normalize = normalize

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        stem_norm = build_norm_layer(normalize, 64)
        self.stem_norm_name = 'gn1' if normalize['type'] == 'GN' else 'bn1'
        self.add_module(self.stem_norm_name, stem_norm)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                normalize=normalize)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

            # zero init for last norm layer https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    last_norm = getattr(m, m.norm_names[-1])
                    constant_init(last_norm, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = getattr(self, self.stem_norm_name)(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.normalize['type'] == 'BN' and self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
 
            stem_norm = getattr(self, self.stem_norm_name)
            stem_norm.eval()
            for param in stem_norm.parameters():
                param.requires_grad = False

            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False


class ResNetClassifier(ResNet):
    def __init__(self,
                 depth,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 normalize=dict(
                     type='BN',
                     frozen_stages=-1,
                     bn_eval=True,
                     bn_frozen=False),
                 with_cp=False,
                 num_classes=1000):
        super(ResNetClassifier, self).__init__(depth,
                                               num_stages=num_stages,
                                               strides=strides,
                                               dilations=dilations,
                                               out_indices=out_indices,
                                               style=style,
                                               normalize=normalize,
                                               with_cp=with_cp)
        _, self.stage_blocks = self.arch_settings[depth]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        expansion = 1 if depth == 18 else 4
        self.fc = nn.Linear(512 * expansion, num_classes)

        self.init_weights()

    # TODO can be removed after tested
    def load_caffe2_weight(self, cf_path):
        norm = 'gn' if self.normalize['type'] == 'GN' else 'bn'
        mapping = {}

        for layer, blocks_in_layer in enumerate(self.stage_blocks, 1):
            for blk in range(blocks_in_layer):
                cf_prefix = 'res%d_%d_' % (layer + 1, blk)
                py_prefix = 'layer%d.%d.' % (layer, blk)

                # conv branch
                for i, a in zip([1, 2, 3], ['a', 'b', 'c']):
                    cf_full = cf_prefix + 'branch2%s_' % a
                    mapping[py_prefix + 'conv%d.weight' % i] = cf_full + 'w'
                    mapping[py_prefix + norm + '%d.weight' % i] \
                        = cf_full + norm + '_s'
                    mapping[py_prefix + norm + '%d.bias' % i] \
                        = cf_full + norm + '_b'

            # downsample branch
            cf_full = 'res%d_0_branch1_' % (layer + 1)
            py_full = 'layer%d.0.downsample.' % layer
            mapping[py_full + '0.weight'] = cf_full + 'w'
            mapping[py_full + '1.weight'] = cf_full + norm + '_s'
            mapping[py_full + '1.bias'] = cf_full + norm + '_b'

        # stem layers and last fc layer
        if self.normalize['type'] == 'GN':
            mapping['conv1.weight'] = 'conv1_w'
            mapping['gn1.weight'] = 'conv1_gn_s'
            mapping['gn1.bias'] = 'conv1_gn_b'
            mapping['fc.weight'] = 'pred_w'
            mapping['fc.bias'] = 'pred_b'
        else:
            mapping['conv1.weight'] = 'conv1_w'
            mapping['bn1.weight'] = 'res_conv1_bn_s'
            mapping['bn1.bias'] = 'res_conv1_bn_b'
            mapping['fc.weight'] = 'fc1000_w'
            mapping['fc.bias'] = 'fc1000_b'

        # load state dict
        py_state = self.state_dict()
        with open(cf_path, 'rb') as f:
            cf_state = pickle.load(f, encoding='latin1')
            if 'blobs' in cf_state:
                cf_state = cf_state['blobs']
            for i, (py_k, cf_k) in enumerate(mapping.items(), 1):
                print('[{}/{}] Loading {} to {}'.format(i, len(mapping), cf_k, py_k))
                assert py_k in py_state and cf_k in cf_state
                py_state[py_k] = torch.Tensor(cf_state[cf_k])
        self.load_state_dict(py_state)

    def forward(self, x):
        x = self.conv1(x)
        x = getattr(self, self.stem_norm_name)(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
