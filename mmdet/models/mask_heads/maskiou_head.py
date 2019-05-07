import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..registry import HEADS


@HEADS.register_module
class MaskIoUHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 num_classes=81):
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = (self.in_channels + 1
                           if i == 0 else self.conv_out_channels)
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    3,
                    stride=stride,
                    padding=1))

        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = self.conv_out_channels * (
                roi_feat_size // 2)**2 if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.mask_iou = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.mask_iou, std=0.01)

    def forward(self, x, mask_pred):
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))
        x = torch.cat((x, mask_pred_pooled), 1)
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.mask_iou(x)
        return mask_iou

    def get_target(self, mask_pred, mask_targets):
        mask_pred = (mask_pred > 0.5).float()  # binarize mask pred
        mask_overlaps = (mask_pred * mask_targets).sum((-1, -2))
        mask_unions = (mask_pred + mask_targets).clamp(max=1.).sum((-1, -2))
        mask_iou_targets = mask_overlaps / mask_unions
        return mask_iou_targets

    def loss(self, mask_iou_pred, mask_iou_targets):
        loss = dict()
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss['loss_mask_iou'] = 0.5 * F.mse_loss(
                mask_iou_pred[pos_inds], mask_iou_targets[pos_inds])[None]
        else:
            loss['loss_mask_iou'] = mask_iou_pred * 0
        return loss

    def set_mask_confidence(self, det_bboxes):
        pass
