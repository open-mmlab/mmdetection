import math

import torch
from mmcv.cnn import ConvModule, xavier_init, normal_init
from torch import nn
from mmcv.runner.base_module import BaseModule

class CARAFE(BaseModule):
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
                groups=1,padding_mode='reflect')
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
                act_cfg=None,
                padding_mode='reflect')
            self.kernel_normalizer.add_module(name='kernel_normalizer_downsample',
                                              module=nn.Conv2d(self.kernel_channel,
                                                               self.kernel_channel,
                                                               kernel_size=k_enc,
                                                               stride=scale,
                                                               padding=(k_enc - 1) // 2,
                                                               groups=1,
                                                               padding_mode='reflect'
                                                               ))
            self.kernel_normalizer.add_module(name='kernel_normalizer_norm', module=nn.Softmax(dim=1))
            self.feature_reassemble.add_module(name='feature_reassemble_downsample',
                                               module=nn.MaxPool2d(kernel_size=2, stride=2))

        self.feature_reassemble.add_module(name='feature_reassemble_unfold',
                                           module=nn.Unfold(kernel_size=k_up, dilation=scale,
                                                            padding=k_up // 2 * scale))


    #     self.init_weights()
    #
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.enc, std=0.001)

    def forward(self, X):
        b, c, h, w = X.size()
        if self.op == 'upsample':
            h_, w_ = h * self.scale, w * self.scale
        else:
            h_, w_ = math.ceil(h / self.scale), math.ceil(w / self.scale)

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w

        W = self.kernel_normalizer(W)
        X = self.feature_reassemble(X)
        X = X.view(b, c, -1, h_, w_)  # b * c * 25 * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X


if __name__ == '__main__':
    x = torch.Tensor(1, 16, 24, 24)
    # carafe = CARAFE(16,c_mid=16,op='downsample')
    carafe = CARAFE(16, c_mid=64, op='upsample')
    oup = carafe(x)
    print(oup.size())
