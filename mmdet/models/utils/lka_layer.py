import  torch.nn as nn
from mmcv.cnn import ConvModule


class AttentionModule(nn.Module):
    '''
        Large Kernel Attention Layer
        https://arxiv.org/abs/2202.09741.pdf
    '''
    def __init__(self, dim ,
                 kernel_size=7,
                 dilation=3,
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # padding = ((kernel_size - 1) * dilation) / 2 保持特征图分辨率不变,stride=1的情况下
        # 空洞卷积的等效卷积核大小尺寸计算公式如下：
        # K=k+(k−1)∗(r−1)
        # 其中，K代表等效卷积核尺寸，k代表实际卷积核尺寸，而r代表dilation，空洞卷积的参数。
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=(kernel_size-1)* dilation//2, groups=dim, dilation=dilation)
        # self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv1 = ConvModule(in_channels=dim,out_channels=dim,kernel_size=1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
    def forward(self, x):
        # u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        # return u * attn
        return attn