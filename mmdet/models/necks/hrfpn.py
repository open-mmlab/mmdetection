import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..registry import NECKS


@NECKS.register_module
class HRFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 pooling='AVG',
                 share_conv=False,
                 with_cp=False):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.share_conv = share_conv
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        if self.share_conv:
            self.fpn_conv = nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3, padding=1)
        else:
            self.fpn_conv = nn.ModuleList()
            for i in range(5):
                self.fpn_conv.append(nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ))
        if pooling == 'MAX':
            print("Using AVG Pooling")
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d
        self.with_cp = with_cp

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(inputs[i],
                                      scale_factor=2**i,
                                      mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, 5):
            outs.append(self.pooling(out, kernel_size=2**i,
                                     stride=2**i))
        outputs = []
        if self.share_conv:
            for i in range(5):
                outputs.append(self.fpn_conv(outs[i]))
        else:
            for i in range(5):
                if outs[i].requires_grad and self.with_cp:
                    tmp_out = checkpoint(self.fpn_conv[i], outs[i])
                else:
                    tmp_out = self.fpn_conv[i](outs[i])
                outputs.append(tmp_out)
        return tuple(outputs)
