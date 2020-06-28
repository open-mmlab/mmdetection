import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, constant_init

from ..builder import BACKBONES
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 rfp_inplanes=None,
                 sac=None,
                 **kwargs):
        """Bottleneck block for DetectoRS_ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        assert sac is None or isinstance(sac, dict)
        self.sac = sac
        self.with_sac = sac is not None
        if self.with_sac:
            self.conv2 = build_conv_layer(
                self.sac,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                bias=False)

        self.rfp_inplanes = rfp_inplanes
        if self.rfp_inplanes:
            self.rfp_conv = build_conv_layer(
                None,
                self.rfp_inplanes,
                planes * self.expansion,
                1,
                stride=1,
                bias=True)
        self.init_weights()

    def init_weights(self):
        if self.rfp_inplanes:
            constant_init(self.rfp_conv, 0)

    def rfp_forward(self, x, rfp_feat):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.rfp_inplanes:
            rfp_feat = self.rfp_conv(rfp_feat)
            out = out + rfp_feat

        out = self.relu(out)

        return out


@BACKBONES.register_module(name='DetectoRS_ResNet')
class DetectoRS_ResNet(ResNet):
    """ResNet backbone for DetectoRS.

    Args:
        sac (dict): Dictionary to construct sac layers.
        stage_with_sac (list): Which stage to use sac.
        rfp_inplanes (int): The number of channels from rfp.
        output_img (bool): If `True`, insert image into the output.
        pretrained (str): the pretrained model to load.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 sac=None,
                 stage_with_sac=(False, False, False, False),
                 rfp_inplanes=None,
                 output_img=False,
                 pretrained=None,
                 **kwargs):
        self.sac = sac
        self.stage_with_sac = stage_with_sac
        self.rfp_inplanes = rfp_inplanes
        self.output_img = output_img
        self.pretrained = pretrained
        super(DetectoRS_ResNet, self).__init__(**kwargs)

        self.inplanes = self.stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            sac = self.sac if self.stage_with_sac[i] else None
            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                sac=sac,
                rfp_inplanes=rfp_inplanes if i > 0 else None,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def forward(self, x):
        """Forward function"""
        outs = list(super(DetectoRS_ResNet, self).forward(x))
        if self.output_img:
            outs.insert(0, x)
        return tuple(outs)

    def rfp_forward(self, x, rfp_feats):
        """Forward function for RFP"""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rfp_feat = rfp_feats[i] if i > 0 else None
            for layer in res_layer:
                x = layer.rfp_forward(x, rfp_feat)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
