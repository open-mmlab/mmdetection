import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

from ..registry import HEADS
from mmdet.core import grid_target


@HEADS.register_module
class GridHead(nn.Module):

    def __init__(self,
                 num_convs=8,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 deconv_kernel=4,
                 num_grids=9,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None):
        super(GridHead, self).__init__()
        self.num_convs = num_convs
        self.roi_feat_size = roi_feat_size  # WARN: not used and reserved
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_grids = num_grids
        self.test_mode = False

        self.convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            strides = 2 if i == 0 else 1
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.conv_out_channels,
                              self.conv_kernel_size, strides, padding),
                    nn.GroupNorm(36, self.conv_out_channels),
                    nn.ReLU(inplace=True)))
        self.convs = nn.Sequential(*self.convs)

        planes = self.conv_out_channels
        self.single_plane = self.conv_out_channels // num_grids

        self.updeconv1 = nn.ConvTranspose2d(
            planes,
            planes,
            kernel_size=deconv_kernel,
            stride=2,
            padding=(deconv_kernel - 2) // 2,
            groups=num_grids)
        self.sbn1 = nn.GroupNorm(num_grids, planes)
        self.updeconv2 = nn.ConvTranspose2d(
            planes,
            num_grids,
            kernel_size=deconv_kernel,
            stride=2,
            padding=(deconv_kernel - 2) // 2,
            groups=num_grids)

        self.neighborpoint = ((1, 3), (0, 2, 4), (1, 5), (0, 4, 6),
                              (1, 3, 5, 7), (2, 4, 8), (3, 7), (4, 6, 8), (5,
                                                                           7))
        self.num_edges = functools.reduce(
            lambda x, y: x + y, map(lambda x: len(x), self.neighborpoint))
        self.firstOrderConvs = []
        self.secondOrderConvs = []
        for _point in self.neighborpoint:
            _foc = [
                nn.Sequential(
                    nn.Conv2d(
                        self.single_plane,
                        self.single_plane,
                        5,
                        1,
                        2,
                        groups=self.single_plane),
                    nn.Conv2d(self.single_plane, self.single_plane, 1, 1, 0))
                for _idx in range(len(_point))
            ]
            _soc = [
                nn.Sequential(
                    nn.Conv2d(
                        self.single_plane,
                        self.single_plane,
                        5,
                        1,
                        2,
                        groups=self.single_plane),
                    nn.Conv2d(self.single_plane, self.single_plane, 1, 1, 0))
                for _idx in range(len(_point))
            ]
            self.firstOrderConvs.append(nn.Sequential(*_foc))
            self.secondOrderConvs.append(nn.Sequential(*_soc))

        self.firstOrderConvs = nn.Sequential(*self.firstOrderConvs)
        self.secondOrderConvs = nn.Sequential(*self.secondOrderConvs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.constant_(self.updeconv2.bias, -np.log(0.99 / 0.01))

    def forward(self, x):
        x = self.convs(x)

        first_order_x = [None] * self.num_grids
        for _idx, _point_idx in enumerate(self.neighborpoint):
            first_order_x[_idx] = x[:, _idx * self.single_plane:(_idx + 1) *
                                    self.single_plane]
            for _iidx, _neighbor_idx in enumerate(_point_idx):
                first_order_x[_idx] = first_order_x[
                    _idx] + self.firstOrderConvs[_idx][_iidx](
                        x[:, _neighbor_idx * self.single_plane:
                          (_neighbor_idx + 1) * self.single_plane])

        second_order_x = [None] * self.num_grids
        for _idx, _point_idx in enumerate(self.neighborpoint):
            second_order_x[_idx] = x[:, _idx * self.single_plane:(_idx + 1) *
                                     self.single_plane]
            for _iidx, _neighbor_idx in enumerate(_point_idx):
                second_order_x[_idx] = second_order_x[
                    _idx] + self.secondOrderConvs[_idx][_iidx](
                        first_order_x[_neighbor_idx])

        x2 = torch.cat(second_order_x, dim=1)
        x2 = self.updeconv1(x2)
        x2 = nn.functional.relu(self.sbn1(x2), inplace=True)
        x2 = self.updeconv2(x2)

        if not self.test_mode:
            x1 = x
            x1 = self.updeconv1(x1)
            x1 = nn.functional.relu(self.sbn1(x1), inplace=True)
            x1 = self.updeconv2(x1)
        else:
            x1 = x2

        return x1, x2

    def get_target(self, sampling_results, rcnn_train_cfg):
        grid_targets = grid_target(sampling_results, rcnn_train_cfg)
        return grid_targets

    def loss(self, grid_pred1, grid_pred2, grid_targets):
        loss = dict()
        grid_loss = F.binary_cross_entropy_with_logits(
            grid_pred1, grid_targets) + F.binary_cross_entropy_with_logits(
                grid_pred2, grid_targets)
        grid_loss = grid_loss * 15

        loss['loss_grid'] = grid_loss
        return loss

    def get_bboxes(self, det_bboxes, grid_pred, img_meta):
        assert (det_bboxes.shape[0] == grid_pred.shape[0])
        det_bboxes = det_bboxes.cpu()
        cls_scores = det_bboxes[:, [4]]
        det_bboxes = det_bboxes[:, :4]
        grid_pred = torch.sigmoid(grid_pred).cpu()

        # expand pos_bboxes
        widths = det_bboxes[:, 2] - det_bboxes[:, 0]
        heights = det_bboxes[:, 3] - det_bboxes[:, 1]
        x1 = det_bboxes[:, 0] - widths / 2
        y1 = det_bboxes[:, 1] - heights / 2

        R, C, H, W = grid_pred.shape
        grid_pred = grid_pred.view(R * C, H * W)
        pred_scores, pred_position = grid_pred.max(dim=1)

        xs = pred_position % W
        ys = pred_position // W
        base = (0, 14, 28)
        for i in range(9):
            xs[i::9] = xs[i::9] + base[i // 3]
            ys[i::9] = ys[i::9] + base[i % 3]
        pred_scores, xs, ys = tuple(
            map(lambda x: x.view(R, C), [pred_scores, xs, ys]))

        grid_points = (xs.float() + 0.5) / (2*W) * widths.view(-1, 1) * 2 + \
            x1.view(-1, 1), \
            (ys.float() + 0.5) / (2*H) * heights.view(-1, 1) * 2 + \
            y1.view(-1, 1)

        x1_idx, y1_idx, x2_idx, y2_idx = ([0, 1, 2], [0, 3, 6],
                                          [6, 7, 8], [2, 5, 8])
        res_dets_x1 = (grid_points[0][:, x1_idx] * pred_scores[:, x1_idx]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x1_idx].sum(dim=1, keepdim=True))
        res_dets_y1 = (grid_points[1][:, y1_idx] * pred_scores[:, y1_idx]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y1_idx].sum(dim=1, keepdim=True))
        res_dets_x2 = (grid_points[0][:, x2_idx] * pred_scores[:, x2_idx]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, x2_idx].sum(dim=1, keepdim=True))
        res_dets_y2 = (grid_points[1][:, y2_idx] * pred_scores[:, y2_idx]).sum(
            dim=1, keepdim=True) / (
                pred_scores[:, y2_idx].sum(dim=1, keepdim=True))

        det_res = torch.cat(
            [res_dets_x1, res_dets_y1, res_dets_x2, res_dets_y2, cls_scores],
            dim=1)
        det_res[:, [0, 2]].clamp_(min=0, max=img_meta[0]['img_shape'][1] - 1)
        det_res[:, [1, 3]].clamp_(min=0, max=img_meta[0]['img_shape'][0] - 1)

        return det_res
