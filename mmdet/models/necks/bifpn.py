import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS


class SingleBiFPN(BaseModule):
    """Single layer of BiFPN.

    Args:
       in_channels (List[int]): Number of input channels per scale.
       out_channels (int): Number of output channels (used at each scale).
       conv_cfg (dict): Config dict for convolution layer.
       norm_cfg (dict): Config dict for normalization layer.
       act_cfg (dict): Config dict for activation layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(SingleBiFPN, self).__init__()
        self.out_channels = out_channels

        if len(in_channels) == 5:
            self.nodes = [
                {
                    'feat_level': 3,
                    'inputs_offsets': [3, 4]
                },
                {
                    'feat_level': 2,
                    'inputs_offsets': [2, 5]
                },
                {
                    'feat_level': 1,
                    'inputs_offsets': [1, 6]
                },
                {
                    'feat_level': 0,
                    'inputs_offsets': [0, 7]
                },
                {
                    'feat_level': 1,
                    'inputs_offsets': [1, 7, 8]
                },
                {
                    'feat_level': 2,
                    'inputs_offsets': [2, 6, 9]
                },
                {
                    'feat_level': 3,
                    'inputs_offsets': [3, 5, 10]
                },
                {
                    'feat_level': 4,
                    'inputs_offsets': [4, 11]
                },
            ]
        elif len(in_channels) == 3:
            self.nodes = [
                {
                    'feat_level': 1,
                    'inputs_offsets': [1, 2]
                },
                {
                    'feat_level': 0,
                    'inputs_offsets': [0, 3]
                },
                {
                    'feat_level': 1,
                    'inputs_offsets': [1, 3, 4]
                },
                {
                    'feat_level': 2,
                    'inputs_offsets': [2, 5]
                },
            ]
        else:
            raise NotImplementedError

        node_info = in_channels.copy()
        for fnode in self.nodes:
            feat_level = fnode['feat_level']
            inputs_offsets = fnode['inputs_offsets']
            inputs_offsets_str = '_'.join(map(str, inputs_offsets))
            for input_offset in inputs_offsets:
                in_channels = node_info[input_offset]
                if in_channels != out_channels:
                    self.add_module(
                        f'lateral_{input_offset}_f{feat_level}',
                        ConvModule(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg))
            node_info.append(out_channels)

            # generate attention weights
            setattr(
                self, f'weights_f{feat_level}_{inputs_offsets_str}',
                nn.Parameter(
                    torch.ones(len(inputs_offsets), dtype=torch.float32),
                    requires_grad=True))

            # generate convolutions after combination
            self.add_module(
                f'outputs_f{feat_level}_{inputs_offsets_str}',
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, feats):
        feats = feats.copy()
        num_levels = len(feats)
        for fnode in self.nodes:
            feat_level = fnode['feat_level']
            inputs_offsets = fnode['inputs_offsets']
            inputs_offsets_str = '_'.join(map(str, inputs_offsets))
            input_nodes = []
            _, _, target_h, target_w = feats[feat_level].size()
            for input_offset in inputs_offsets:
                input_node = feats[input_offset]
                # reduction
                if input_node.size(1) != self.out_channels:
                    input_node = getattr(
                        self, f'lateral_{input_offset}'
                        f'_f{feat_level}')(
                            input_node)
                # resize
                _, _, h, w = input_node.size()
                if h > target_h and w > target_w:
                    height_stride_size = int((h - 1) // target_h + 1)
                    width_stride_size = int((w - 1) // target_w + 1)
                    assert height_stride_size == width_stride_size == 2
                    input_node = F.max_pool2d(
                        input_node, kernel_size=3, stride=2, padding=1)
                elif h <= target_h and w <= target_w:
                    if h < target_h or w < target_w:
                        input_node = F.interpolate(
                            input_node,
                            size=(target_h, target_w),
                            mode='nearest')
                else:
                    raise NotImplementedError
                input_nodes.append(input_node)
            # attention
            weights = F.relu(
                getattr(self, f'weights_f{feat_level}_{inputs_offsets_str}'))
            norm_weights = weights / (weights.sum() + 0.0001)
            new_node = torch.stack(input_nodes, dim=-1)
            new_node = (norm_weights * new_node).sum(dim=-1)
            new_node = new_node * new_node.sigmoid()
            feats.append(
                getattr(self, f'outputs_f{feat_level}'
                        f'_{inputs_offsets_str}')(new_node))

        output_feats = []
        for idx in range(num_levels):
            for i, fnode in enumerate(reversed(self.nodes)):
                if fnode['feat_level'] == idx:
                    output_feats.append(feats[-1 - i])
                    break
            else:
                raise ValueError()
        return output_feats


@NECKS.register_module()
class BiFPN(BaseModule):
    """The neck proposed in 'EfficientDet https://arxiv.org/abs/1911.09070'_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales (implemented case 3 and 5).
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (inclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        num_bifpn (int): Number of bifpn layers.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.

     Example:
        >>> import torch
        >>> from mmdet.models.necks import BiFPN
        >>> in_channels = [64, 128, 256]
        >>> scales = [8, 16, 32]
        >>> num_outs = 5
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = BiFPN(in_channels, 256, num_outs).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 256, 8, 8])
        outputs[1].shape = torch.Size([1, 256, 16, 16])
        outputs[2].shape = torch.Size([1, 256, 32, 32])
        outputs[3].shape = torch.Size([1, 256, 16, 16])
        outputs[4].shape = torch.Size([1, 256, 8, 8])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 num_bifpn=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                 )):
        super(BiFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        assert num_outs == 5 or num_outs == 3, NotImplementedError
        self.num_bifpn = num_bifpn
        self.extra_levels = nn.ModuleList()
        self.start_level = start_level
        self.end_level = \
            end_level if end_level != -1 else len(in_channels) - 1

        end_level = self.end_level
        num_extra_levels = num_outs - (end_level - start_level + 1)
        in_channel = in_channels[end_level]
        for i in range(num_extra_levels):
            if in_channel != out_channels:
                extra_level = nn.Sequential(
                    ConvModule(
                        in_channel,
                        out_channels,
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

            else:
                extra_level = torch.nn.MaxPool2d(
                    kernel_size=3, stride=2, padding=1)
            in_channel = out_channels
            self.extra_levels.append(extra_level)

        self.repeated_bifpn = nn.ModuleList()
        for i in range(num_bifpn):
            if i == 0:
                in_channels = in_channels[start_level:end_level + 1] + \
                                   [out_channels] * num_extra_levels
            else:
                in_channels = [out_channels] * num_outs
            self.repeated_bifpn.append(
                SingleBiFPN(
                    in_channels,
                    out_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        laterals = [
            inputs[i] for i in range(self.start_level, self.end_level + 1)
        ]
        if len(self.extra_levels) > 0:
            for extra_level in self.extra_levels:
                laterals.append(extra_level(laterals[-1]))
        for i in range(self.num_bifpn):
            laterals = self.repeated_bifpn[i](laterals)
        outs = tuple(laterals)
        return outs
