import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS


@NECKS.register_module
class BIFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack = 1, # [JW]
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 # [JW] norm 여부 => True면 안함, ConvModule에서 이 값이 bias가 'auto'일 때 bias 유무도 결정
                 no_norm_on_lateral=False,
                 # [JW] 여기서는 실제로 conv_cfg를 사용하는 경우는 없음. 즉, 항상 None
                 conv_cfg=None,
                 # [JW] norm_cfg ex) dict(type='BN', requires_grad=False)
                 norm_cfg=None,
                 act_cfg=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.stack = stack # [JW]
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

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
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList() # [JW]

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # [JW] 원하는 num_outs 보다 (backbone_end_level - start_level) 이 짧을 경우
        #      fpn_convs (extra conv layers)를 통해 backbone이 되는 부분 길이 확장
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # [JW] stack_bifpn_conv layers
        for i in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(
                                            channels=out_channels,
                                            levels=self.backbone_end_level - start_level,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            activation=activation))

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    # [JW] fp16 이 의미하는 것 무엇..?
    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # [JW] original code of FPN
        # # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     prev_shape = laterals[i - 1].shape[2:]
        #     laterals[i - 1] += F.interpolate(
        #         laterals[i], size=prev_shape, mode='nearest')
        
        # # build outputs
        # # part 1: from original levels
        # outs = [
        #     self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        # ]

        # [JW] BiFPN part
        used_backbone_levels = len(laterals)
        for bifpn in self.stack_bifpn_convs:
            laterals = bifpn(laterals)
        outs = laterals
        
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                # [JW] 기존의 input에 대해 conv 진행해서 추가
                if self.extra_convs_on_inputs:
                    # [JW] 기존 있는 level 중 가장 상위 level
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    # BiFPN의 마지막 결과에 대해 conv 진행해서 추가
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BiFPNModule, self).__init__()
        self.levels = levels
        self.activation = activation
        self.bifpn_convs = nn.ModuleList()

        # [JW] weights for weighted sum
        # [JW] ex) w for P_td_6 in paper
        self.w1 = nn.Parameters(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        # [JW] ex) w for P_out_6 in paper
        self.w2 = nn.Parameters(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU() # [JW] relu 굳이 1,2로 나누는 이유?, F.relu()와 차이?

        for i in range(2):
            for j in range(levels - 1): # [JW] 총 levels-1 * 2 개 (아래 forward에서 총 2*levels-2개의 bifpn_conv 사용)
                bifpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        groups=channels, # [JW] seperable conv => channel 별 conv.. 맞나요?
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=activation,
                        inplace=False),
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=activation,
                        inplace=False))
                bifpn_convs.append(bifpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16
    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and bottom-up path with stack
        levels = self.levels

        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + eps
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + eps

        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        # build top-down
        kk = 0
        for i in range(levels - 1, 0, -1): # [JW] paper로 치면 level 6 -> 3
            # [JW] 참고했던 코드에선 inputs_clone[i - 1] 대신 pathtd[i - 1] 로 했는데,
            #      논문 의미 그대로 구현하기 위해 inputs_clone으로 사용
            pathtd[i - 1] = w1[0, kk] * inputs_clone[i - 1] + \
                            w1[1, kk] * F.interpolate(inputs_clone[i], scale_factor=2)
            pathtd[i - 1] = self.bifpn_convs[kk](pathtd[i - 1])
            kk += 1

        # build bottom-up
        jj = kk
        for i in range(levels - 2): # [JW] paper로 치면 level4 -> 6
            pathtd[i + 1] = w2[0, i] * inputs_clone[i + 1] + \
                            w2[1, i] * pathtd[i + 1] + \
                            w2[2, i] * F.max_pool2d(inputs_clone[i], kernel_size=2)
            pathtd[i + 1] = self.bifpn_convs[jj](pathtd[i + 1])
            jj += 1
        # [JW] 참고했던 코드에선 inputs_clone[i - 1] 대신 pathtd[i - 1] 로 했는데,
        #      논문 의미 그대로 구현하기 위해 inputs_clone으로 사용
        pathtd[levels - 1] = w1[0, kk] * inputs_clone[levels - 1] + \
                             w1[1, kk] * F.max_pool2d(pathtd[levels - 2], kernel_size=2)
        pathtd[levels - 1] = self.bifpn_convs[jj](pathtd[levels - 1]) # [JW] paper로 치면 level 7

        return pathtd






