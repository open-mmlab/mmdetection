from mmcv.cnn import ConvModule

class ConvModule_Norm(ConvModule):
    def __init__(self, in_channels,
                out_channels,
                kernel, **kwargs):
        super().__init__(in_channels, out_channels, kernel, **kwargs)

        self.normType = kwargs.get('norm_cfg', {'type':''})
        if self.normType is not None:
            self.normType = self.normType['type']
    
    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                if 'LN' in self.normType:
                    x = x.permute(0, 2, 3, 1)
                    x = self.norm(x)
                    x = x.permute(0, 3, 1, 2).contiguous()
                else:
                    x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x
