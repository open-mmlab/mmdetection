# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from torch import Tensor
import math
from mmdet.registry import MODELS
from .atss_head import ATSSHead
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms
from mmengine.structures import InstanceData
from mmdet.structures.bbox import scale_boxes
from mmdet.structures import DetDataSample

from transformers import BertConfig
import torch.utils.checkpoint as checkpoint
from typing import List, Optional, Tuple
from mmdet.utils import InstanceList, OptMultiConfig
from mmengine.config import ConfigDict
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)
import copy
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                              groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DYReLU(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        visual_feats = inputs["visual"]
        language_dict_features = inputs["lang"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args))
            if level < len(visual_feats) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](visual_feats[level + 1], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features}

        return features_dict


from mmengine.model import BaseModel


# TODO: move VLFusion related Classes to a separate file
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        bsz, tgt_len, embed_dim = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):

        visu_feat = []
        size_per_level, visual_features_flatten = [], []
        for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
            bs, c, h, w = feat_per_level.shape
            size_per_level.append([h, w])
            feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
            visual_features_flatten.append(feat)
        visual_features_flatten = torch.cat(visual_features_flatten, dim=1)
        new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l)
        # [bs, N, C] -> [bs, C, N]
        new_v = new_v.transpose(1, 2).contiguous()

        start = 0
        for (h, w) in size_per_level:
            new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
            visu_feat.append(new_v_per_level)
            start += h * w

        lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], \
            lang_feat[2], lang_feat[3], lang_feat[4]

    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self):
        super(VLFuse, self).__init__()
        self.init_configs()

        # early fusion module
        # TODO: support use_checkpoint as cfg
        self.use_checkpoint = True
        if self.use_checkpoint:  # True
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        print("EARLY FUSION ON, USING {}".format("MHA-B"))

        # bi-direction (text->image, image->text)
        self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                                                    l_dim=self.lang_dim,
                                                    embed_dim=self.embed_dim,
                                                    num_heads=self.n_head,
                                                    hidden_dim=self.i2t_hidden_dim,
                                                    dropout=0.1,
                                                    drop_path=.0,
                                                    init_values=1.0 / 6.0
                                                    )

    def init_configs(self):
        # common params
        self.lang_model = "bert-base-uncased"
        self.joint_embedding_size = 256
        self.joint_embedding_dropout = 0.1
        self.joint_mlp_layers = 2

        self.max_query_len = 256
        self.n_layers = 1
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = 256

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        self.lang_dim = 768

    def forward(self, x):
        # import pdb; pdb.set_trace()
        visual_features = x["visual"]
        language_dict_features = x["lang"]

        batch_size = visual_features[0].shape[0]
        device = visual_features[0].device

        fused_visual_features = None
        fused_language_dict_features = None

        if self.use_checkpoint:
            q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                                                                           visual_features[0], visual_features[1],
                                                                           visual_features[2], visual_features[3],
                                                                           visual_features[4],
                                                                           language_dict_features['hidden'],
                                                                           language_dict_features['masks'],
                                                                           self.dummy_tensor
                                                                           )
        else:
            q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                visual_features[0], visual_features[1],
                visual_features[2], visual_features[3],
                visual_features[4],
                language_dict_features['hidden'],
                language_dict_features['masks'],
                self.dummy_tensor
            )

        # q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
        #     visual_features[0], visual_features[1],
        #     visual_features[2], visual_features[3],
        #     visual_features[4],
        #     language_dict_features['hidden'],
        #     language_dict_features['masks']
        # )

        fused_visual_features = [q0, q1, q2, q3, q4]
        language_features = l0

        language_dict_features['hidden'] = language_features
        fused_language_dict_features = language_dict_features

        features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features}

        return features_dict


class VLFusionModule(BaseModel):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_base_priors,
                 num_classes,
                 early_fuse=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_base_priors = num_base_priors
        self.num_classes = num_classes
        self.early_fuse = early_fuse
        self._init_layers()

    def _init_layers(self) -> None:
        use_dyrelu = True
        use_dyfuse = True
        use_deform = True
        bn_type = ['gn', 16]
        num_dyhead_blocks = 6
        conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)
        log_scale = 0.0
        prior_prob = 0.01
        lang_dim = 768

        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # TODO: put the name into kwargs or cfg
        lang_cfg = BertConfig.from_pretrained('bert-base-uncased')

        dyhead_tower = []
        for i in range(num_dyhead_blocks):
            if self.early_fuse:
                # cross-modality fusion
                dyhead_tower.append(
                    # TODO: add init parameters for VLFUSE
                    VLFuse()
                )
                # self language path
                from ..language_models import BertEncoderLayer
                dyhead_tower.append(
                    BertEncoderLayer(
                        lang_cfg,
                        clamp_min_for_underflow=True,
                        clamp_max_for_overflow=True)
                )

            dyhead_tower.append(
                DyConv(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and self.in_channels == self.feat_channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and self.in_channels == self.feat_channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and self.in_channels == self.feat_channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(self.feat_channels, self.num_base_priors * self.num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, kernel_size=1)  # num_anchors=1
        self.centerness = nn.Conv2d(self.feat_channels, self.num_base_priors * 1, kernel_size=1)

        self.dot_product_projection_image = nn.Identity()
        # 将语言模型输出进行投影到视觉语义上
        self.dot_product_projection_text = nn.Linear(lang_dim,
                                                     self.num_base_priors * self.feat_channels, bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        # DEBUG
        # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict):
        logits = []
        bbox_reg = []
        centerness = []

        feat_inputs = {"visual": visual_feats,
                       "lang": language_feats}

        dyhead_tower = self.dyhead_tower(feat_inputs)

        dot_product_logits = []

        if self.early_fuse:
            embedding = dyhead_tower["lang"]["hidden"]
        else:
            embedding = language_feats['embedded']

        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # text embeding (1,256,768)

        # 语言特征投影到视觉空间
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)  # (1,256,256)
        # print(embedding.sum(), dot_product_proj_tokens.sum())
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0  # (1, 256)

        for l, feature in enumerate(visual_feats):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))  # (1,80,100,136)

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(dyhead_tower["visual"][l]))

            x = dyhead_tower["visual"][l]
            B, C, H, W = x.shape

            # add bias (language)
            # 图像特征作为 query，文本特征作为 key，计算相似度
            dot_product_proj_queries = self.dot_product_projection_image(x)
            dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)  # 1,13600,256

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            # dot_product_proj_tokens 融合后的文本特征 1,13600,256
            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1,
                                                                                                          -2)) / self.log_scale.exp()) + bias
            # print(x.sum(), dot_product_logit.sum(), dot_product_proj_queries.sum(), dot_product_proj_tokens.sum(),
            #       self.log_scale)

            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            dot_product_logits.append(dot_product_logit)

        return logits, bbox_reg, centerness, dot_product_logits


def convert_grounding_to_od_logits(logits, box_cls, positive_maps, score_agg=None):
    # 这个 scores 维度是 (1,13600,80)，这个 80 是明显不合理的，这个地方应该是当前句子中 token 的个数
    # 假设当前句子一共 3 个命名实体，那么这个维度应该是 (1,13600,3)
    # 虽然结果一样，但是含义就不一样，当某一种图片的实体超过 80 那就会报错了
    assert len(positive_maps) == logits.shape[0]

    scores = torch.zeros(logits.shape[0], logits.shape[1], box_cls.shape[-1]).to(logits.device)  # (1,13600,80)
    # 256 -> 80, average for each class
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            # score aggregation method
            if score_agg == "MEAN":  # ture
                for label_j in positive_map:  # logits (1,13600,256) 取出对应 token 位置的预测值，然后求均值,将其转换为 80 类的预测值
                    scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].mean(-1)
            elif score_agg == "MAX":
                # torch.max() returns (values, indices)
                for label_j in positive_map:
                    scores[:, :, label_j - 1] = logits[:, :, torch.LongTensor(positive_map[label_j])].max(-1)[
                        0]
            elif score_agg == "ONEHOT":
                # one hot
                scores = logits[:, :, :len(positive_map)]
            else:
                raise NotImplementedError
        else:
            for i, positive_map in enumerate(positive_maps):
                if score_agg == "MEAN":  # ture
                    for label_j in positive_map:  # logits (1,13600,256) 取出对应 token 位置的预测值，然后求均值,将其转换为 80 类的预测值
                        scores[i, :, label_j - 1] = logits[i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
                elif score_agg == "MAX":
                    # torch.max() returns (values, indices)
                    for label_j in positive_map:
                        scores[i, :, label_j - 1] = logits[i, :, torch.LongTensor(positive_map[label_j])].max(-1)[
                            0]
                elif score_agg == "ONEHOT":
                    # one hot
                    raise NotImplementedError
                else:
                    raise NotImplementedError

    return scores


@MODELS.register_module()
class ATSSVLFusionHead(ATSSHead):
    def __init__(self, *args, early_fuse=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = VLFusionModule(in_channels=self.in_channels,
                                   feat_channels=self.feat_channels,
                                   num_base_priors=self.num_base_priors,
                                   num_classes=self.num_classes,
                                   early_fuse=early_fuse)

    def _init_layers(self) -> None:
        pass

    def forward(self, visual_feats: Tuple[Tensor], language_feats: dict, ):
        cls_scores, bbox_preds, centerness, dot_product_logits = self.head(
            visual_feats,
            language_feats
        )
        return cls_scores, bbox_preds, centerness, dot_product_logits

    def predict(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict,
                batch_data_samples,
                rescale: bool = True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map for data_samples in batch_data_samples
        ]
        outs = self(visual_feats, language_feats)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: List[Tensor],
                        dot_product_logits: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds) == len(score_factors)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            token_positive_map = batch_token_positive_maps[img_id]
            # 实际上 cls_score_list 不需要
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            score_factor_list = select_single_mlvl(
                score_factors, img_id, detach=True)
            dot_product_logit_list = select_single_mlvl(dot_product_logits, img_id, detach=True)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                dot_product_logit_list=dot_product_logit_list,
                mlvl_priors=mlvl_priors,
                token_positive_map=token_positive_map,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                dot_product_logit_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                token_positive_map: dict,
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, bbox_pred, score_factor, dot_product_logit, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, dot_product_logit_list, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            score_factor = score_factor.permute(1, 2,
                                                0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            dot_product_logit = dot_product_logit.sigmoid()
            # TODO cls_score 不需要
            scores = convert_grounding_to_od_logits(logits=dot_product_logit[None], box_cls=scores,
                                                    positive_maps=[token_positive_map],
                                                    score_agg="MEAN")[0]
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))

            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            score_factor = score_factor[keep_idxs]

            scores = torch.sqrt(scores * score_factor)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        predictions = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)
        # important
        predictions.labels = predictions.labels + 1
        return predictions
