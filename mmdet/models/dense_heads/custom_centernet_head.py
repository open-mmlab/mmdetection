import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

from .centernet_head import CenterNetHead

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@HEADS.register_module()
class Custom_CenterNetHead(CenterNetHead):
    def __init__(self):
        super(Custom_CenterNetHead, self).__init__()
        # todo: initialize the
        # cls_tower
        # bbox_tower
        # agn_hm
        # bbox_pred
        #


        self.cls_tower()



        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channels if i == 0 else channel,
                        channel,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0)])
        # self.scales = nn.ModuleList(
        #     [Scale(init_value=1.0) for _ in input_shape])


    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)


    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            bbox_reg(Tensor): reg predicts, the channels number is 4
            agn_hms (Tensor): center predict heatmaps, the channels number is 1
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred

        #feature = self.share_tower(feature)
        cls_tower = self.cls_tower(feat)
        bbox_tower = self.bbox_tower(feat)

        # if not self.only_proposal:
        #     clss.append(self.cls_logits(cls_tower))
        # else:
        #     clss.append(None)
        agn_hms = self.agn_hm(bbox_tower)
        reg = self.bbox_pred(bbox_tower)
        reg = self.scales[0](reg)
        # reg = self.scales[l](reg)
        bbox_reg = F.relu(reg)

    return bbox_reg, agn_hms