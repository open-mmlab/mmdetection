# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import caffe2_xavier_init, kaiming_init
from torch.nn import init

from mmdet.registry import MODELS


def _make_stack_3x3_convs(num_convs,
                          in_channels,
                          out_channels,
                          act_cfg=dict(type='ReLU', inplace=True)):
    convs = []
    for _ in range(num_convs):
        convs.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(MODELS.build(act_cfg))
        in_channels = out_channels
    return nn.Sequential(*convs)


class InstanceBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_convs=4,
                 num_masks=100,
                 num_classes=80,
                 kernel_dim=128,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        num_masks = num_masks
        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim,
                                                act_cfg)
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

        # outputs
        self.cls_score = nn.Linear(dim, self.num_classes)
        self.mask_kernel = nn.Linear(dim, kernel_dim)
        self.objectness = nn.Linear(dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob,
                                  features.view(B, C, -1).permute(0, 2, 1))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


class MaskBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 dim=256,
                 num_convs=4,
                 kernel_dim=128,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim,
                                                act_cfg)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        kaiming_init(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


@MODELS.register_module()
class BaseIAMDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 ins_dim=256,
                 ins_conv=4,
                 mask_dim=256,
                 mask_conv=4,
                 kernel_dim=128,
                 scale_factor=2.0,
                 output_iam=False,
                 num_masks=100,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels  # ENCODER.NUM_CHANNELS + 2

        self.scale_factor = scale_factor
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(
            in_channels,
            dim=ins_dim,
            num_convs=ins_conv,
            num_masks=num_masks,
            num_classes=num_classes,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)
        self.mask_branch = MaskBranch(
            in_channels,
            dim=mask_dim,
            num_convs=mask_conv,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
        mask_features = self.mask_branch(features)

        N = pred_kernel.shape[1]
        # mask_features: BxCxHxW
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel,
                               mask_features.view(B, C,
                                                  H * W)).view(B, N, H, W)

        pred_masks = F.interpolate(
            pred_masks,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False)

        output = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
        }

        if self.output_iam:
            iam = F.interpolate(
                iam,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False)
            output['pred_iam'] = iam

        return output


class GroupInstanceBranch(nn.Module):

    def __init__(self,
                 in_channels,
                 num_groups=4,
                 dim=256,
                 num_convs=4,
                 num_masks=100,
                 num_classes=80,
                 kernel_dim=128,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.num_groups = num_groups
        self.num_classes = num_classes

        self.inst_convs = _make_stack_3x3_convs(
            num_convs, in_channels, dim, act_cfg=act_cfg)
        # iam prediction, a group conv
        expand_dim = dim * self.num_groups
        self.iam_conv = nn.Conv2d(
            dim,
            num_masks * self.num_groups,
            3,
            padding=1,
            groups=self.num_groups)
        # outputs
        self.fc = nn.Linear(expand_dim, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        caffe2_xavier_init(self.fc)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)
        iam_prob = iam.sigmoid()

        B, N = iam_prob.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob,
                                  features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(B, 4, N // self.num_groups,
                                              -1).transpose(1, 2).reshape(
                                                  B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@MODELS.register_module()
class GroupIAMDecoder(BaseIAMDecoder):

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_groups=4,
                 ins_dim=256,
                 ins_conv=4,
                 mask_dim=256,
                 mask_conv=4,
                 kernel_dim=128,
                 scale_factor=2.0,
                 output_iam=False,
                 num_masks=100,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            ins_dim=ins_dim,
            ins_conv=ins_conv,
            mask_dim=mask_dim,
            mask_conv=mask_conv,
            kernel_dim=kernel_dim,
            scale_factor=scale_factor,
            output_iam=output_iam,
            num_masks=num_masks,
            act_cfg=act_cfg)
        self.inst_branch = GroupInstanceBranch(
            in_channels,
            num_groups=num_groups,
            dim=ins_dim,
            num_convs=ins_conv,
            num_masks=num_masks,
            num_classes=num_classes,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)


class GroupInstanceSoftBranch(GroupInstanceBranch):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax_bias = nn.Parameter(torch.ones([
            1,
        ]))

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N = iam.shape[:2]
        C = features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(iam_prob,
                                  features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(B, self.num_groups,
                                              N // self.num_groups,
                                              -1).transpose(1, 2).reshape(
                                                  B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc(inst_features))
        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        return pred_logits, pred_kernel, pred_scores, iam


@MODELS.register_module()
class GroupIAMSoftDecoder(BaseIAMDecoder):

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_groups=4,
                 ins_dim=256,
                 ins_conv=4,
                 mask_dim=256,
                 mask_conv=4,
                 kernel_dim=128,
                 scale_factor=2.0,
                 output_iam=False,
                 num_masks=100,
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            ins_dim=ins_dim,
            ins_conv=ins_conv,
            mask_dim=mask_dim,
            mask_conv=mask_conv,
            kernel_dim=kernel_dim,
            scale_factor=scale_factor,
            output_iam=output_iam,
            num_masks=num_masks,
            act_cfg=act_cfg)
        self.inst_branch = GroupInstanceSoftBranch(
            in_channels,
            num_groups=num_groups,
            dim=ins_dim,
            num_convs=ins_conv,
            num_masks=num_masks,
            num_classes=num_classes,
            kernel_dim=kernel_dim,
            act_cfg=act_cfg)
