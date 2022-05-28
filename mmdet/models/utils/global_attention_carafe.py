import math

import torch
from mmcv.cnn import ConvModule, xavier_init, normal_init, kaiming_init
from torch import nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.cbam import SpatialGate

class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)
    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 2, 5, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2]).contiguous()
        out = torch.cat(out, dim=1)
        return out


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c,
                 op='upsample',
                 c_mid=64,
                 scale=2,
                 k_up=5,
                 k_enc=3
                 ):
        """ The unofficial implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        assert op in ['downsample', 'upsample']
        super(CARAFE, self).__init__()
        self.scale = scale

        self.c_mid = c_mid
        self.op = op
        self.comp = nn.Conv2d(c, c_mid, kernel_size=1)
        self.kernel_normalizer = nn.Sequential()
        self.feature_reassemble = nn.Sequential()

        if self.op == 'upsample':
            self.kernel_channel = (scale * k_up) ** 2
            self.enc = nn.Conv2d(
                c_mid,
                self.kernel_channel,
                kernel_size=k_enc,
                stride=1, padding=k_enc // 2, dilation=1,
                groups=1)
            self.kernel_normalizer.add_module(name='kernel_normalizer_upsample', module=nn.PixelShuffle(scale))
            self.kernel_normalizer.add_module(name='kernel_normalizer_norm', module=nn.Softmax(dim=1))
            self.feature_reassemble.add_module(name='feature_reassemble_upsample',
                                               module=nn.Upsample(scale_factor=scale, mode='bilinear'))

        elif self.op == 'downsample':
            self.kernel_channel = k_up ** 2
            self.enc = ConvModule(
                c_mid,
                self.kernel_channel,
                kernel_size=k_enc,
                stride=1, padding=k_enc // 2, dilation=1,
                groups=1,
                act_cfg=None)
            self.kernel_normalizer.add_module(name='kernel_normalizer_downsample',
                                              module=nn.Conv2d(self.kernel_channel,
                                                               self.kernel_channel,
                                                               kernel_size=k_enc,
                                                               stride=scale,
                                                               padding=(k_enc - 1) // 2,
                                                               groups=1,
                                                               padding_mode='replicate'
                                                               ))
            self.kernel_normalizer.add_module(name='kernel_normalizer_norm', module=nn.Softmax(dim=1))
            self.feature_reassemble.add_module(name='feature_reassemble_downsample',
                                               module=nn.MaxPool2d(kernel_size=2, stride=2))

        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, X):
        b, c, h, w = X.size()
        if self.op == 'upsample':
            h_, w_ = h * self.scale, w * self.scale
        else:
            h_, w_ = math.ceil(h / self.scale), math.ceil(w / self.scale)

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w

        W = self.kernel_normalizer(W) # b * 25 * h_ * w_
        # W = self.pix_shf(W)
        # W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.feature_reassemble(X)
        X = self.unfold(X).contiguous()  # b * c * h_ * w_ -> b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_).contiguous()  # b * c * 25 * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


class Attention_CARAFE(BaseModule):
    def __init__(self, op, in_channel):
        assert op in ['downsample', 'upsample']
        super(Attention_CARAFE, self).__init__()
        self.in_channel = in_channel
        self.op = op
        self.ratio = 16
        att_chn = 2 * in_channel

        carafe_mid_channel = 16 if op == 'downsample' else 64
        # self.CARAFE = CARAFE(att_chn, c_mid=carafe_mid_channel, op=self.op)
        self.CARAFE = nn.Sequential(
            CARAFE(att_chn, c_mid=carafe_mid_channel, op=self.op),
            nn.Conv2d(att_chn,in_channel,kernel_size=1)
        )
        if op == 'upsample':
            # self.trans = nn.UpsamplingBilinear2d(scale_factor=2)
            self.trans = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # self.trans = nn.MaxPool2d(kernel_size=2, stride=2)
            self.trans = nn.UpsamplingBilinear2d(scale_factor=2)
        norm_cfg = dict(type='BN')
        act_cfg = dict(type='ReLU')

        # self.spatial_att = nn.Sequential(nn.Conv2d(att_chn, att_chn,
        #                                            kernel_size=3, stride=1, padding=1),
        #                                  ASPP(att_chn, att_chn // 4),
        #                                  ConvModule(att_chn, att_chn,
        #                                             kernel_size=3, stride=1, padding=1,
        #                                             norm_cfg=norm_cfg, act_cfg=act_cfg),
        #                                  nn.Conv2d(att_chn, 1, kernel_size=1),
        #                                  nn.Softmax()
        #                                  )
        # self.channel_att = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                                  ConvModule(
        #                                      in_channels=att_chn,
        #                                      out_channels=int(att_chn / self.ratio),
        #                                      kernel_size=1,
        #                                      stride=1,
        #                                      conv_cfg=None,
        #                                      act_cfg=act_cfg),
        #                                  ConvModule(
        #                                      in_channels=int(att_chn / self.ratio),
        #                                      out_channels=att_chn,
        #                                      kernel_size=1,
        #                                      stride=1,
        #                                      conv_cfg=None,
        #                                      act_cfg=None),
        #                                  nn.Sigmoid()
        #                                  )
        self.spatial_att = SpatialGate()

        self.channel_att = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         ConvModule(
                                             in_channels=att_chn,
                                             out_channels=int(att_chn / self.ratio),
                                             kernel_size=1,
                                             stride=1,
                                             conv_cfg=None,
                                             act_cfg=act_cfg),
                                         ConvModule(
                                             in_channels=int(att_chn / self.ratio),
                                             out_channels=att_chn,
                                             kernel_size=1,
                                             stride=1,
                                             conv_cfg=None,
                                             act_cfg=None),
                                         nn.Sigmoid()
                                         )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, lower, upper):
        lower_identity = lower.clone()
        upper_identity = upper.clone()

        if self.op == 'upsample':
            upper = self.trans(upper)
        else:
            lower = self.trans(lower)

        mid = torch.cat((lower, upper), dim=1)  # 1 * 2c * h * w
        mid_clone = mid.clone()

        # att = self.spatial_att(mid)
        # mid = att * mid
        mid = self.spatial_att(mid)

        mid = self.channel_att(mid)
        lower_att, upper_att = torch.split(mid, self.in_channel, dim=1)

        mid_clone = self.CARAFE(mid_clone)

        if self.op == 'upsample':
            upper_result = upper_identity * upper_att
            lower_result = mid_clone * lower_att
        else:
            lower_result = lower_identity * lower_att
            upper_result = mid_clone * upper_att

        return lower_result + upper_result

def get_module(op,in_channel):
    assert op in ['upsample','downsample']
    return Attention_CARAFE(op=op,in_channel=in_channel)



if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    # carafe = CARAFE(16,c_mid=16,op='downsample')
    carafe = CARAFE(16, c_mid=64, op='upsample')
    oup = carafe(x)
    # model = Attention_CARAFE()
    # oup =
    print(oup.size())
