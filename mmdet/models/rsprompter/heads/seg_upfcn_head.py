import einops
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.models import build_loss
from mmdet.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
import torch.nn.functional as F


@MODELS.register_module()
class UpFCNHead(BaseModule):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 mid_channels=[256, 128, 64],
                 num_classes=2,
                 kernel_size=3,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 align_corners=False,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.align_corners = align_corners

        if isinstance(in_channels, list):
            self.pre_layers = nn.ModuleList()
            inner_channel = mid_channels[0]
            for idx, channel in enumerate(in_channels):
                self.pre_layers.append(
                    nn.Sequential(
                        ConvModule(
                            channel,
                            inner_channel,
                            kernel_size=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                        ),
                        ConvModule(
                            inner_channel,
                            inner_channel,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                        ),
                    )
                )
            self.pre_layers.append(
                nn.Sequential(
                    ConvModule(
                        inner_channel*len(in_channels),
                        inner_channel,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        inner_channel,
                        inner_channel,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )
            input_channel = inner_channel
        else:
            input_channel = in_channels

        convs = []
        for idx, mid_channel in enumerate(mid_channels):
            in_channel = input_channel if idx == 0 else mid_channels[idx-1]
            convs += [
                ConvModule(
                    in_channel,
                    mid_channel,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    mid_channel,
                    mid_channel,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                nn.UpsamplingBilinear2d(scale_factor=2),
            ]
        self.convs = nn.Sequential(*convs)
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        self.conv_seg = nn.Conv2d(mid_channels[-1], num_classes, kernel_size=1)

    def _forward_feature(self, img_feat, inner_states):
        if hasattr(self, 'pre_layers'):
            inner_states = inner_states[-len(self.in_channels):]
            inner_states = [einops.rearrange(x, 'b h w c -> b c h w') for x in inner_states]
            inner_states = [layer(x) for layer, x in zip(self.pre_layers[:-1], inner_states)]
            img_feat = self.pre_layers[-1](torch.cat(inner_states, dim=1))
        feats = self.convs(img_feat)
        return feats

    def forward(self, img_feat, inner_states):
        """Forward function."""
        output = self._forward_feature(img_feat, inner_states)
        output = self.conv_seg(output)
        return output

    def loss(self, img_feat, inner_states, batch_data_samples) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(img_feat, inner_states)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def _stack_batch_gt(self, batch_data_samples):
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits, batch_data_samples) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        losses = dict()
        seg_logits = F.interpolate(seg_logits, seg_label.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        seg_label = seg_label.squeeze(1)
        losses['loss_ce'] = self.loss_decode(seg_logits, seg_label)
        return losses

    def predict(self, img_feat, inner_states):
        seg_logits = self.forward(img_feat, inner_states)
        return seg_logits
