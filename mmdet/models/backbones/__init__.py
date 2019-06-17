from .resnet import ResNet, make_res_layer
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .hrnet import HRNet
from .ssd_vgg_m2det import SSDVGG_M2Det
__all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet','SSDVGG_M2Det']
