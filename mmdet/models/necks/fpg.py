import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init, constant_init, is_norm

from ..builder import NECKS


class Transition(nn.Module):
    """Base class for transition.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(x):
        pass


class UpInterpolationConv(Transition):
    """A transition used for up-sampling.

    Up-sample the input by interpolation then refines the feature by
    a convolution layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Up-sampling factor. Default: 2.
        mode (int): Interpolation mode. Default: nearest.
        align_corners (bool): Whether align corners when interpolation.
            Default: None.
        kernel_size (int): Kernel size for the conv. Default: 3.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=2,
                 mode='nearest',
                 align_corners=None,
                 kernel_size=3,
                 **kwargs):
        super().__init__(in_channels, out_channels)
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            **kwargs)

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        x = self.conv(x)
        return x


class LastConv(Transition):
    """A transition used for refining the output of the last stage.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_inputs (int): Number of inputs of the FPN features.
        kernel_size (int): Kernel size for the conv. Default: 3.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_inputs,
                 kernel_size=3,
                 **kwargs):
        super().__init__(in_channels, out_channels)
        self.num_inputs = num_inputs
        self.conv_out = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            **kwargs)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs
        return self.conv_out(inputs[-1])


@NECKS.register_module()
class FPG(nn.Module):
    """FPG.

    Implementation of `Feature Pyramid Grids (FPG)
    <https://arxiv.org/abs/2004.03580>`_.
    This implementation only gives the basic structure stated in the paper.
    But users can implement different type of transitions to fully explore the
    the potential power of the structure of FPG.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        stack_times (int): The number of times the pyramid architecture will
            be stacked.
        paths (list[str]): Specify the path order of each stack level.
            Each element in the list should be either 'bu' (bottom-up) or
            'td' (top-down).
        inter_channels (int): Number of inter channels.
        same_up_trans (dict): Transition that goes down at the same stage.
        same_down_trans (dict): Transition that goes up at the same stage.
        across_lateral_trans (dict): Across-pathway same-stage
        across_down_trans (dict): Across-pathway bottom-up connection.
        across_up_trans (dict): Across-pathway top-down connection.
        across_skip_trans (dict): Across-pathway skip connection.
        output_trans (dict): Transition that trans the output of the
            last stage.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): It decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    transition_types = {
        'conv': ConvModule,
        'interpolation_conv': UpInterpolationConv,
        'last_conv': LastConv,
    }

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 stack_times,
                 paths,
                 inter_channels=None,
                 same_down_trans=None,
                 same_up_trans=dict(
                     type='conv', kernel_size=3, stride=2, padding=1),
                 across_lateral_trans=dict(type='conv', kernel_size=1),
                 across_down_trans=dict(type='conv', kernel_size=3),
                 across_up_trans=None,
                 across_skip_trans=dict(type='identity'),
                 output_trans=dict(type='last_conv', kernel_size=3),
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 norm_cfg=None,
                 skip_inds=None):
        super(FPG, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        if inter_channels is None:
            self.inter_channels = [out_channels for _ in range(num_outs)]
        elif isinstance(inter_channels, int):
            self.inter_channels = [inter_channels for _ in range(num_outs)]
        else:
            assert isinstance(inter_channels, list)
            assert len(inter_channels) == num_outs
            self.inter_channels = inter_channels
        self.stack_times = stack_times
        self.paths = paths
        assert isinstance(paths, list) and len(paths) == stack_times
        for d in paths:
            assert d in ('bu', 'td')

        self.same_down_trans = same_down_trans
        self.same_up_trans = same_up_trans
        self.across_lateral_trans = across_lateral_trans
        self.across_down_trans = across_down_trans
        self.across_up_trans = across_up_trans
        self.output_trans = output_trans
        self.across_skip_trans = across_skip_trans

        self.with_bias = norm_cfg is None
        # skip inds must be specified if across skip trans is not None
        if self.across_skip_trans is not None:
            skip_inds is not None
        self.skip_inds = skip_inds
        assert len(self.skip_inds[0]) <= self.stack_times

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        # build lateral 1x1 convs to reduce channels
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(self.in_channels[i],
                               self.inter_channels[i - self.start_level], 1)
            self.lateral_convs.append(l_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            if self.add_extra_convs:
                fpn_idx = self.backbone_end_level - self.start_level + i
                extra_conv = nn.Conv2d(
                    self.inter_channels[fpn_idx - 1],
                    self.inter_channels[fpn_idx],
                    3,
                    stride=2,
                    padding=1)
                self.extra_downsamples.append(extra_conv)
            else:
                self.extra_downsamples.append(nn.MaxPool2d(1, stride=2))

        self.fpn_transitions = nn.ModuleList()  # stack times
        for s in range(self.stack_times):
            stage_trans = nn.ModuleList()  # num of feature levels
            for i in range(self.num_outs):
                # same, across_lateral, across_down, across_up
                trans = nn.ModuleDict()
                if s in self.skip_inds[i]:
                    stage_trans.append(trans)
                    continue
                # build same-stage down trans (used in bottom-up paths)
                if i == 0 or self.same_up_trans is None:
                    same_up_trans = None
                else:
                    same_up_trans = self.build_trans(
                        self.same_up_trans, self.inter_channels[i - 1],
                        self.inter_channels[i])
                trans['same_up'] = same_up_trans
                # build same-stage up trans (used in top-down paths)
                if i == self.num_outs - 1 or self.same_down_trans is None:
                    same_down_trans = None
                else:
                    same_down_trans = self.build_trans(
                        self.same_down_trans, self.inter_channels[i + 1],
                        self.inter_channels[i])
                trans['same_down'] = same_down_trans
                # build across lateral trans
                across_lateral_trans = self.build_trans(
                    self.across_lateral_trans, self.inter_channels[i],
                    self.inter_channels[i])
                trans['across_lateral'] = across_lateral_trans
                # build across down trans
                if i == self.num_outs - 1 or self.across_down_trans is None:
                    across_down_trans = None
                else:
                    across_down_trans = self.build_trans(
                        self.across_down_trans, self.inter_channels[i + 1],
                        self.inter_channels[i])
                trans['across_down'] = across_down_trans
                # build across up trans
                if i == 0 or self.across_up_trans is None:
                    across_up_trans = None
                else:
                    across_up_trans = self.build_trans(
                        self.across_up_trans, self.inter_channels[i - 1],
                        self.inter_channels[i])
                trans['across_up'] = across_up_trans
                if self.across_skip_trans is None:
                    across_skip_trans = None
                else:
                    across_skip_trans = self.build_trans(
                        self.across_skip_trans, self.inter_channels[i - 1],
                        self.inter_channels[i])
                trans['across_skip'] = across_skip_trans
                # build across_skip trans
                stage_trans.append(trans)
            self.fpn_transitions.append(stage_trans)

        self.output_transition = nn.ModuleList()  # output levels
        for i in range(self.num_outs):
            trans = self.build_trans(
                self.output_trans,
                self.inter_channels[i],
                self.out_channels,
                num_inputs=self.stack_times + 1)
            self.output_transition.append(trans)

        self.relu = nn.ReLU(inplace=True)

    def build_trans(self, cfg, in_channels, out_channels, **extra_args):
        cfg_ = cfg.copy()
        trans_type = cfg_.pop('type')
        trans_cls = self.transition_types[trans_type]
        return trans_cls(in_channels, out_channels, **cfg_, **extra_args)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)
            elif is_norm(m):
                constant_init(m, 1.0)

    def fuse(self, fuse_dict):
        out = None
        for item in fuse_dict.values():
            if item is not None:
                if out is None:
                    out = item
                else:
                    out = out + item
        return out

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build all levels from original feature maps
        feats = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for downsample in self.extra_downsamples:
            feats.append(downsample(feats[-1]))

        outs = [feats]

        for i in range(self.stack_times):
            current_outs = outs[-1]
            next_outs = []
            direction = self.paths[i]
            for j in range(self.num_outs):
                if i in self.skip_inds[j]:
                    next_outs.append(outs[-1][j])
                    continue
                # feature level
                if direction == 'td':
                    lvl = self.num_outs - j - 1
                else:
                    lvl = j
                # get transitions
                if direction == 'td':
                    same_trans = self.fpn_transitions[i][lvl]['same_down']
                else:
                    same_trans = self.fpn_transitions[i][lvl]['same_up']
                across_lateral_trans = self.fpn_transitions[i][lvl][
                    'across_lateral']
                across_down_trans = self.fpn_transitions[i][lvl]['across_down']
                across_up_trans = self.fpn_transitions[i][lvl]['across_up']
                across_skip_trans = self.fpn_transitions[i][lvl]['across_skip']
                # init output
                to_fuse = dict(
                    same=None, lateral=None, across_up=None, across_down=None)
                # same downsample/upsample
                if same_trans is not None:
                    to_fuse['same'] = same_trans(next_outs[-1])
                # across lateral
                if across_lateral_trans is not None:
                    to_fuse['lateral'] = across_lateral_trans(
                        current_outs[lvl])
                # across downsample
                if lvl > 0 and across_up_trans is not None:
                    to_fuse['across_up'] = across_up_trans(current_outs[lvl -
                                                                        1])
                # across upsample
                if (lvl < self.num_outs - 1 and across_down_trans is not None):
                    to_fuse['across_down'] = across_down_trans(
                        current_outs[lvl + 1])
                if across_skip_trans is not None:
                    to_fuse['across_skip'] = across_skip_trans(outs[0][lvl])
                x = self.fuse(to_fuse)
                next_outs.append(x)

            if direction == 'td':
                outs.append(next_outs[::-1])
            else:
                outs.append(next_outs)

        # output trans
        final_outs = []
        for i in range(self.num_outs):
            lvl_out_list = []
            for s in range(len(outs)):
                lvl_out_list.append(outs[s][i])
            lvl_out = self.output_transition[i](lvl_out_list)
            final_outs.append(lvl_out)

        return final_outs
