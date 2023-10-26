from mmdet.utils import InstanceList
from mmengine.config import ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn import DepthwiseSeparableConvModule
from .base_dense_head import BaseDenseHead

from mmdet.registry import MODELS, TASK_UTILS

from typing import List, Optional, Tuple, Sequence, Dict
from torch import Tensor

import copy
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm, DepthwiseSeparableConvModule
from mmengine.model import bias_init_with_prob, constant_init, normal_init, BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList)
from ..task_modules.samplers import PseudoSampler
from ..utils import filter_scores_and_topk, images_to_levels, multi_apply
from .base_dense_head import BaseDenseHead

# def dist2bbox(distance, anchor_points, box_format='xyxy'):
#     '''Transform distance(ltrb) to box(xywh or xyxy).'''
#     lt, rb = torch.split(distance, 2, -1)
#     x1y1 = anchor_points - lt
#     x2y2 = anchor_points + rb
#     if box_format == 'xyxy':
#         bbox = torch.cat([x1y1, x2y2], -1)
#     elif box_format == 'xywh':
#         c_xy = (x1y1 + x2y2) / 2
#         wh = x2y2 - x1y1
#         bbox = torch.cat([c_xy, wh], -1)
#     return bbox

# def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):
#     '''Generate anchors from features.'''
#     anchors = []
#     anchor_points = []
#     stride_tensor = []
#     num_anchors_list = []
#     assert feats is not None
#     if is_eval:
#         for i, stride in enumerate(fpn_strides):
#             _, _, h, w = feats[i].shape
#             shift_x = torch.arange(end=w, device=device) + grid_cell_offset
#             shift_y = torch.arange(end=h, device=device) + grid_cell_offset
#             shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
#             anchor_point = torch.stack(
#                     [shift_x, shift_y], axis=-1).to(torch.float)
#             if mode == 'af': # anchor-free
#                 anchor_points.append(anchor_point.reshape([-1, 2]))
#                 stride_tensor.append(
#                 torch.full(
#                     (h * w, 1), stride, dtype=torch.float, device=device))
#             elif mode == 'ab': # anchor-based
#                 anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
#                 stride_tensor.append(
#                     torch.full(
#                         (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
#         anchor_points = torch.cat(anchor_points)
#         stride_tensor = torch.cat(stride_tensor)
#         return anchor_points, stride_tensor
#     else:
#         for i, stride in enumerate(fpn_strides):
#             _, _, h, w = feats[i].shape
#             cell_half_size = grid_cell_size * stride * 0.5
#             shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
#             shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
#             shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
#             anchor = torch.stack(
#                 [
#                     shift_x - cell_half_size, shift_y - cell_half_size,
#                     shift_x + cell_half_size, shift_y + cell_half_size
#                 ],
#                 axis=-1).clone().to(feats[0].dtype)
#             anchor_point = torch.stack(
#                 [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

#             if mode == 'af': # anchor-free
#                 anchors.append(anchor.reshape([-1, 4]))
#                 anchor_points.append(anchor_point.reshape([-1, 2]))
#             elif mode == 'ab': # anchor-based
#                 anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
#                 anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
#             num_anchors_list.append(len(anchors[-1]))
#             stride_tensor.append(
#                 torch.full(
#                     [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
#         anchors = torch.cat(anchors)
#         anchor_points = torch.cat(anchor_points).to(device)
#         stride_tensor = torch.cat(stride_tensor).to(device)
#         return anchors, anchor_points, num_anchors_list, stride_tensor

class YOLOV6LiteSubHeadNetwork(BaseModule):
    def __init__(self, in_channels, num_anchors, num_classes):
        
        self.stem = DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            group=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict('Hardswish')
        )
        
        self.cls_conv = DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            group=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict('Hardswish')
        )
        
        self.reg_conv = DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            group=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict('Hardswish')
        )
        
        self.cls_pred = ConvModule(
            in_channels=in_channels,
            out_channels=num_classes * num_anchors,
            kernel_size=1,
            stride=1,
            padding=0,
            group=1,
            norm_cfg=None,
            act_cfg=None
        )
        
        self.reg_pred = ConvModule(
            in_channels=in_channels,
            out_channels=4 * num_anchors,
            kernel_size=1,
            stride=1,
            padding=0,
            group=1,
            norm_cfg=None,
            act_cfg=None
        )
        
    def forward(self, x):
        x = self.stem(x)
        reg = self.reg_conv(x)
        reg = self.reg_pred(reg)
        clas = self.cls_conv(x)
        clas = self.cls_pred(clas)
        return (clas, reg)
    
class YOLOV6LiteHeadNetwork(BaseModule):
    def __init__(self, in_channels, num_anchors, num_classes):
        # process the 0th pyramid layer (the most shallow features)
        self.subhead0 = YOLOV6LiteSubHeadNetwork(
            in_channels=in_channels[0],
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        # layer 1
        self.subhead1 = YOLOV6LiteSubHeadNetwork(
            in_channels=in_channels[1],
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        # layer 2
        self.subhead2 = YOLOV6LiteSubHeadNetwork(
            in_channels=in_channels[2],
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        # layer 3
        self.subhead3 = YOLOV6LiteSubHeadNetwork(
            in_channels=in_channels[3],
            num_anchors=num_anchors,
            num_classes=num_classes
        )
    
    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Dict[str: Tensor]]:
        
        (fm3, fm2, fm1, fm0) = x
        
        (clas0, reg0) = self.subhead0(fm0)
        (clas1, reg1) = self.subhead0(fm1)
        (clas2, reg2) = self.subhead0(fm2)
        (clas3, reg3) = self.subhead0(fm3)
        
        out = (
            dict(clas=clas3, reg=reg3),
            dict(clas=clas2, reg=reg2),
            dict(clas=clas1, reg=reg1),
            dict(clas=clas0, reg=reg0),
        )
        
        return out

# @MODELS.register_module()
# class YOLOV6LiteHead(BaseDenseHead):
#     def __init__(
#         self,
#         in_channels: Sequence[int],
#         num_anchors: int,
#         num_classes=80,
#     ):
#         self.heads = YOLOV6LiteHeadNetwork(
#             in_channels=in_channels,
#             num_anchors=num_anchors,
#             num_classes=num_classes
#         )
        
        
    
#     def forward(self, x: Tuple[Tensor, ...]) -> tuple:
#         pass
    
#     def predict_by_feat(
#         self,
#         pred_maps: Sequence[Tensor],
#         batch_img_metas: Optional[List[dict]],
#         cfg: OptConfigType = None,
#         rescale: bool = False,
#         with_nms: bool = True
#     ) -> InstanceList:
#         pass
                        
        
        
# class Detect(nn.Module):
#     export = False
#     '''Efficient Decoupled Head
#     With hardware-aware degisn, the decoupled head is optimized with
#     hybridchannels methods.
#     '''
#     def __init__(self, num_classes=80, num_layers=3, head_layers=None):  # detection layer
#         super().__init__()
#         assert head_layers is not None
#         self.nc = num_classes  # number of classes
#         self.no = num_classes + 5  # number of outputs per anchor
#         self.nl = num_layers  # number of detection layers
#         self.grid = [torch.zeros(1)] * num_layers
#         self.prior_prob = 1e-2
#         stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
#         self.stride = torch.tensor(stride)
#         self.grid_cell_offset = 0.5
#         self.grid_cell_size = 5.0

#         # Init decouple head
#         self.stems = nn.ModuleList()
#         self.cls_convs = nn.ModuleList()
#         self.reg_convs = nn.ModuleList()
#         self.cls_preds = nn.ModuleList()
#         self.reg_preds = nn.ModuleList()

#         # Efficient decoupled head layers
#         for i in range(num_layers):
#             idx = i*5
#             self.stems.append(head_layers[idx])
#             self.cls_convs.append(head_layers[idx+1])
#             self.reg_convs.append(head_layers[idx+2])
#             self.cls_preds.append(head_layers[idx+3])
#             self.reg_preds.append(head_layers[idx+4])

#     def initialize_biases(self):

#         for conv in self.cls_preds:
#             b = conv.bias.view(-1, )
#             b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
#             conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#             w = conv.weight
#             w.data.fill_(0.)
#             conv.weight = torch.nn.Parameter(w, requires_grad=True)

#         for conv in self.reg_preds:
#             b = conv.bias.view(-1, )
#             b.data.fill_(1.0)
#             conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#             w = conv.weight
#             w.data.fill_(0.)
#             conv.weight = torch.nn.Parameter(w, requires_grad=True)

#     def forward(self, x):
#         if self.training:
#             cls_score_list = []
#             reg_distri_list = []

#             for i in range(self.nl):
#                 x[i] = self.stems[i](x[i])
#                 cls_x = x[i]
#                 reg_x = x[i]
#                 cls_feat = self.cls_convs[i](cls_x)
#                 cls_output = self.cls_preds[i](cls_feat)
#                 reg_feat = self.reg_convs[i](reg_x)
#                 reg_output = self.reg_preds[i](reg_feat)

#                 cls_output = torch.sigmoid(cls_output)
#                 cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
#                 reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

#             cls_score_list = torch.cat(cls_score_list, axis=1)
#             reg_distri_list = torch.cat(reg_distri_list, axis=1)

#             return x, cls_score_list, reg_distri_list
#         else:
#             cls_score_list = []
#             reg_dist_list = []

#             for i in range(self.nl):
#                 b, _, h, w = x[i].shape
#                 l = h * w
#                 x[i] = self.stems[i](x[i])
#                 cls_x = x[i]
#                 reg_x = x[i]
#                 cls_feat = self.cls_convs[i](cls_x)
#                 cls_output = self.cls_preds[i](cls_feat)
#                 reg_feat = self.reg_convs[i](reg_x)
#                 reg_output = self.reg_preds[i](reg_feat)

#                 cls_output = torch.sigmoid(cls_output)

#                 if self.export:
#                     cls_score_list.append(cls_output)
#                     reg_dist_list.append(reg_output)
#                 else:
#                     cls_score_list.append(cls_output.reshape([b, self.nc, l]))
#                     reg_dist_list.append(reg_output.reshape([b, 4, l]))


#             if self.export:
#                 return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

#             cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
#             reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


#             anchor_points, stride_tensor = generate_anchors(
#                 x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

#             pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
#             pred_bboxes *= stride_tensor
#             return torch.cat(
#                 [
#                     pred_bboxes,
#                     torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
#                     cls_score_list
#                 ],
#                 axis=-1)

# def build_effidehead_layer(channels_list, num_anchors, num_classes, num_layers):

#     head_layers = nn.Sequential(
#         # stem0
#         DPBlock(
#             in_channel=channels_list[0],
#             out_channel=channels_list[0],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_conv0
#         DPBlock(
#             in_channel=channels_list[0],
#             out_channel=channels_list[0],
#             kernel_size=5,
#             stride=1
#         ),
#         # reg_conv0
#         DPBlock(
#             in_channel=channels_list[0],
#             out_channel=channels_list[0],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_pred0
#         nn.Conv2d(
#             in_channels=channels_list[0],
#             out_channels=num_classes * num_anchors,
#             kernel_size=1
#         ),
#         # reg_pred0
#         nn.Conv2d(
#             in_channels=channels_list[0],
#             out_channels=4 * num_anchors,
#             kernel_size=1
#         ),
#         # stem1
#         DPBlock(
#             in_channel=channels_list[1],
#             out_channel=channels_list[1],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_conv1
#         DPBlock(
#             in_channel=channels_list[1],
#             out_channel=channels_list[1],
#             kernel_size=5,
#             stride=1
#         ),
#         # reg_conv1
#         DPBlock(
#             in_channel=channels_list[1],
#             out_channel=channels_list[1],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_pred1
#         nn.Conv2d(
#             in_channels=channels_list[1],
#             out_channels=num_classes * num_anchors,
#             kernel_size=1
#         ),
#         # reg_pred1
#         nn.Conv2d(
#             in_channels=channels_list[1],
#             out_channels=4 * num_anchors,
#             kernel_size=1
#         ),
#         # stem2
#         DPBlock(
#             in_channel=channels_list[2],
#             out_channel=channels_list[2],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_conv2
#         DPBlock(
#             in_channel=channels_list[2],
#             out_channel=channels_list[2],
#             kernel_size=5,
#             stride=1
#         ),
#         # reg_conv2
#         DPBlock(
#             in_channel=channels_list[2],
#             out_channel=channels_list[2],
#             kernel_size=5,
#             stride=1
#         ),
#         # cls_pred2
#         nn.Conv2d(
#             in_channels=channels_list[2],
#             out_channels=num_classes * num_anchors,
#             kernel_size=1
#         ),
#         # reg_pred2
#         nn.Conv2d(
#             in_channels=channels_list[2],
#             out_channels=4 * num_anchors,
#             kernel_size=1
#         )
#     )

#     if num_layers == 4:
#         head_layers.add_module('stem3',
#             # stem3
#             DPBlock(
#                 in_channel=channels_list[3],
#                 out_channel=channels_list[3],
#                 kernel_size=5,
#                 stride=1
#             )
#         )
#         head_layers.add_module('cls_conv3',
#             # cls_conv3
#             DPBlock(
#                 in_channel=channels_list[3],
#                 out_channel=channels_list[3],
#                 kernel_size=5,
#                 stride=1
#             )
#         )
#         head_layers.add_module('reg_conv3',
#             # reg_conv3
#             DPBlock(
#                 in_channel=channels_list[3],
#                 out_channel=channels_list[3],
#                 kernel_size=5,
#                 stride=1
#             )
#         )
#         head_layers.add_module('cls_pred3',
#             # cls_pred3
#             nn.Conv2d(
#                 in_channels=channels_list[3],
#                 out_channels=num_classes * num_anchors,
#                 kernel_size=1
#             )
#          )
#         head_layers.add_module('reg_pred3',
#             # reg_pred3
#             nn.Conv2d(
#                 in_channels=channels_list[3],
#                 out_channels=4 * num_anchors,
#                 kernel_size=1
#             )
#         )

#     return head_layers