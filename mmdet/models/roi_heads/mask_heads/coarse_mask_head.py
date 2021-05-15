from mmcv.cnn import ConvModule, Linear
from mmcv.runner import ModuleList, auto_fp16

from mmdet.models.builder import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module()
class CoarseMaskHead(FCNMaskHead):
    """Coarse mask head used in PointRend.

    Compared with standard ``FCNMaskHead``, ``CoarseMaskHead`` will downsample
    the input feature map instead of upsample it.

    Args:
        num_convs (int): Number of conv layers in the head. Default: 0.
        num_fcs (int): Number of fc layers in the head. Default: 2.
        fc_out_channels (int): Number of output channels of fc layer.
            Default: 1024.
        downsample_factor (int): The factor that feature map is downsampled by.
            Default: 2.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_convs=0,
                 num_fcs=2,
                 fc_out_channels=1024,
                 downsample_factor=2,
                 init_cfg=dict(
                     type='Xavier',
                     override=[
                         dict(name='fcs'),
                         dict(type='Constant', val=0.001, name='fc_logits')
                     ]),
                 *arg,
                 **kwarg):
        super(CoarseMaskHead, self).__init__(
            *arg,
            num_convs=num_convs,
            upsample_cfg=dict(type=None),
            init_cfg=None,
            **kwarg)
        self.init_cfg = init_cfg
        self.num_fcs = num_fcs
        assert self.num_fcs > 0
        self.fc_out_channels = fc_out_channels
        self.downsample_factor = downsample_factor
        assert self.downsample_factor >= 1
        # remove conv_logit
        delattr(self, 'conv_logits')

        if downsample_factor > 1:
            downsample_in_channels = (
                self.conv_out_channels
                if self.num_convs > 0 else self.in_channels)
            self.downsample_conv = ConvModule(
                downsample_in_channels,
                self.conv_out_channels,
                kernel_size=downsample_factor,
                stride=downsample_factor,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        else:
            self.downsample_conv = None

        self.output_size = (self.roi_feat_size[0] // downsample_factor,
                            self.roi_feat_size[1] // downsample_factor)
        self.output_area = self.output_size[0] * self.output_size[1]

        last_layer_dim = self.conv_out_channels * self.output_area

        self.fcs = ModuleList()
        for i in range(num_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            self.fcs.append(Linear(fc_in_channels, self.fc_out_channels))
        last_layer_dim = self.fc_out_channels
        output_channels = self.num_classes * self.output_area
        self.fc_logits = Linear(last_layer_dim, output_channels)

    def init_weights(self):
        super(FCNMaskHead, self).init_weights()

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        if self.downsample_conv is not None:
            x = self.downsample_conv(x)

        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_pred = self.fc_logits(x).view(
            x.size(0), self.num_classes, *self.output_size)
        return mask_pred
