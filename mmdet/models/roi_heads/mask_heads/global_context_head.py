import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16, force_fp32

from mmdet.models.builder import HEADS
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock


@HEADS.register_module()
class GlobalContextHead(nn.Module):
    """Global context head."""

    def __init__(self,
                 num_ins,
                 num_convs=4,
                 in_channels=256,
                 conv_out_channels=256,
                 num_classes=81,
                 loss_weight=1.0,
                 conv_cfg=None,
                 norm_cfg=None,
                 conv_to_res=False):
        super(GlobalContextHead, self).__init__()
        self.num_ins = num_ins
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.conv_to_res = conv_to_res
        self.fp16_enabled = False

        if self.conv_to_res:
            num_res_blocks = num_convs // 2
            self.convs = ResLayer(
                SimplifiedBasicBlock,
                in_channels,
                self.conv_out_channels,
                num_res_blocks,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            self.num_convs = num_res_blocks
        else:
            self.convs = nn.ModuleList()
            for i in range(self.num_convs):
                in_channels = self.in_channels if i == 0 else conv_out_channels
                self.convs.append(
                    ConvModule(
                        in_channels,
                        conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_out_channels, num_classes)

        self.criterion = nn.BCEWithLogitsLoss()

    def init_weights(self):
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    @auto_fp16()
    def forward(self, feats):
        x = feats[-1]
        for i in range(self.num_convs):
            x = self.convs[i](x)
        x = self.pool(x)

        # multi-class prediction
        mc_pred = x.reshape(x.size(0), -1)
        mc_pred = self.fc(mc_pred)

        return mc_pred, x

    @force_fp32(apply_to=('pred', ))
    def loss(self, pred, labels):
        labels = [lbl.unique() for lbl in labels]
        targets = pred.new_zeros(pred.size())
        for i, label in enumerate(labels):
            targets[i, label] = 1.0
        loss = self.loss_weight * self.criterion(pred, targets)
        return loss
