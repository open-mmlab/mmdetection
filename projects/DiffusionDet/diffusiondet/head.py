# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import random
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox2roi, bbox_cxcywh_to_xyxy,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1, ) * (len(x_shape) - 1)))


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@MODELS.register_module()
class DynamicDiffusionDetHead(nn.Module):

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 num_proposals=500,
                 num_heads=6,
                 prior_prob=0.01,
                 snr_scale=2.0,
                 timesteps=1000,
                 sampling_timesteps=1,
                 self_condition=False,
                 box_renewal=True,
                 use_ensemble=True,
                 deep_supervision=True,
                 criterion=dict(
                     type='DiffusionDetCriterion',
                     num_classes=80,
                     assigner=dict(
                         type='DiffusionDetMatcher',
                         match_costs=[
                             dict(
                                 type='FocalLossCost',
                                 alpha=2.0,
                                 gamma=0.25,
                                 weight=2.0),
                             dict(
                                 type='BBoxL1Cost',
                                 weight=5.0,
                                 box_format='xyxy'),
                             dict(type='IoUCost', iou_mode='giou', weight=2.0)
                         ],
                         center_radius=2.5,
                         candidate_topk=5),
                 ),
                 single_head=dict(
                     type='DiffusionDetHead',
                     num_cls_convs=1,
                     num_reg_convs=3,
                     dim_feedforward=2048,
                     num_heads=8,
                     dropout=0.0,
                     act_cfg=dict(type='ReLU'),
                     dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
                 roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 **kwargs) -> None:
        super().__init__()
        self.roi_extractor = MODELS.build(roi_extractor)

        self.num_classes = num_classes
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        # self.filter_empty_ann = filter_empty_ann

        # Build Diffusion
        assert isinstance(timesteps, int), 'The type of `timesteps` should ' \
                                           f'be int but got {type(timesteps)}'
        assert sampling_timesteps <= timesteps
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.snr_scale = snr_scale

        self.ddim_sampling = self.sampling_timesteps < self.timesteps
        self.self_condition = self_condition
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble

        self._build_diffusion()

        # Build assigner
        assert criterion.get('assigner', None) is not None
        assigner = TASK_UTILS.build(criterion.get('assigner'))
        # Init parameters.
        self.use_focal_loss = assigner.use_focal_loss
        self.use_fed_loss = assigner.use_fed_loss

        # build criterion
        criterion.update(deep_supervision=deep_supervision)
        self.criterion = TASK_UTILS.build(criterion)

        # Build Dynamic Head.
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}')
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `DynamicDiffusionDetHead` and '
                    '`SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}')
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size')
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `DynamicDiffusionDetHead` '
                    'and `SingleDiffusionDetHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {num_classes}')
                single_head_.update(
                    pooler_resolution=default_pooler_resolution)

        single_head_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss)
        single_head_module = MODELS.build(single_head_)

        self.num_heads = num_heads
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)])

        self.deep_supervision = deep_supervision

        # Gaussian random feature embedding layer for time
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim))

        self.prior_prob = prior_prob
        self._init_weights()

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or \
                        p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, bias_value)

    def _build_diffusion(self):
        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1. - alphas_cumprod))

    def forward(self, features, init_bboxes, init_t, init_features=None):
        time = self.time_mlp(init_t, )

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = single_head(
                features, bboxes, proposal_features, self.roi_extractor, time)
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return torch.stack(inter_class_logits), torch.stack(
                inter_pred_bboxes)
        else:
            return class_logits[None, ...], pred_bboxes[None, ...]

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        prepare_outputs = self.prepare_targets(batch_data_samples)
        (batch_gt_instances, batch_pred_instances, batch_gt_instances_ignore,
         batch_img_metas) = prepare_outputs

        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes
            for pred_instances in batch_pred_instances
        ])
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])

        pred_logits, pred_bboxes = self(x, batch_diff_bboxes, batch_time)

        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)
        return losses

    def prepare_targets(self, batch_data_samples):
        batch_gt_instances = []
        batch_pred_instances = []
        batch_gt_instances_ignore = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            img_meta = data_sample.metainfo
            gt_instances = data_sample.gt_instances

            gt_bboxes = gt_instances.bboxes
            h, w = img_meta['img_shape']
            image_size = gt_bboxes.new_tensor([w, h, w, h])

            norm_gt_bboxes = gt_bboxes / image_size
            norm_gt_bboxes_cxcywh = bbox_xyxy_to_cxcywh(norm_gt_bboxes)
            pred_instances = self.prepare_diffusion(norm_gt_bboxes_cxcywh,
                                                    image_size)

            gt_instances.set_metainfo(dict(image_size=image_size))
            gt_instances.norm_bboxes = norm_gt_bboxes

            batch_gt_instances.append(gt_instances)
            batch_pred_instances.append(pred_instances)
            batch_img_metas.append(data_sample.metainfo)
            if 'ignored_instances' in data_sample:
                batch_gt_instances_ignore.append(data_sample.ignored_instances)
            else:
                batch_gt_instances_ignore.append(None)
        return (batch_gt_instances, batch_pred_instances,
                batch_gt_instances_ignore, batch_img_metas)

    def prepare_diffusion(self, gt_boxes, image_size):
        device = gt_boxes.device
        time = torch.randint(
            0, self.timesteps, (1, ), dtype=torch.long, device=device)
        noise = torch.randn(self.num_proposals, 4, device=device)

        num_gt = gt_boxes.shape[0]
        if num_gt < self.num_proposals:
            # 3 * sigma = 1/2 --> sigma: 1/6
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, 4, device=device) / 6. + 0.5
            box_placeholder[:, 2:] = torch.clip(
                box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            select_mask = [True] * self.num_proposals + \
                          [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]

        x_start = (x_start * 2. - 1.) * self.snr_scale

        # noise sample
        x = self.q_sample(x_start=x_start, time=time, noise=noise)

        x = torch.clamp(x, min=-1 * self.snr_scale, max=self.snr_scale)
        x = ((x / self.snr_scale) + 1) / 2.

        diff_bboxes = bbox_cxcywh_to_xyxy(x)
        # convert to abs bboxes
        diff_bboxes = diff_bboxes * image_size

        metainfo = dict(time=time.squeeze(-1))
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes
        pred_instances.noise = noise
        return pred_instances

    # forward diffusion
    def q_sample(self, x_start, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_start_shape = x_start.shape

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, time,
                                        x_start_shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, time, x_start_shape)

        return sqrt_alphas_cumprod_t * x_start + \
            sqrt_one_minus_alphas_cumprod_t * noise

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions


@MODELS.register_module()
class SingleDiffusionDetHead(nn.Module):

    def __init__(
        self,
        num_classes=80,
        feat_channels=256,
        dim_feedforward=2048,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=7,
        scale_clamp=_DEFAULT_SCALE_CLAMP,
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        use_focal_loss=True,
        use_fed_loss=False,
        act_cfg=dict(type='ReLU', inplace=True),
        dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels

        # Dynamic
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = build_activation_layer(act_cfg)

        # block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(feat_channels * 4, feat_channels * 2))

        # cls.
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(nn.Linear(feat_channels, feat_channels, False))
            cls_module.append(nn.LayerNorm(feat_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(nn.Linear(feat_channels, feat_channels, False))
            reg_module.append(nn.LayerNorm(feat_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, num_boxes, 4)
        :param pro_features: (N, num_boxes, feat_channels)
        """

        N, num_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])
        rois = bbox2roi(proposal_boxes)

        roi_features = pooler(features, rois)

        if pro_features is None:
            pro_features = roi_features.view(N, num_boxes, self.feat_channels,
                                             -1).mean(-1)

        roi_features = roi_features.view(N * num_boxes, self.feat_channels,
                                         -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, num_boxes,
                                         self.feat_channels).permute(1, 0, 2)
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(
            num_boxes, N,
            self.feat_channels).permute(1, 0,
                                        2).reshape(1, N * num_boxes,
                                                   self.feat_channels)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * num_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, num_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return (class_logits.view(N, num_boxes,
                                  -1), pred_bboxes.view(N, num_boxes,
                                                        -1), obj_features)

    def apply_deltas(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self,
                 feat_channels: int,
                 dynamic_dim: int = 64,
                 dynamic_num: int = 2,
                 pooler_resolution: int = 7) -> None:
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.feat_channels * pooler_resolution**2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, pro_features: Tensor, roi_features: Tensor) -> Tensor:
        """Forward function.

        Args:
            pro_features: (1,  N * num_boxes, self.feat_channels)
            roi_features: (49, N * num_boxes, self.feat_channels)

        Returns:
        """
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim)
        param2 = parameters[:, :,
                            self.num_params:].view(-1, self.dynamic_dim,
                                                   self.feat_channels)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features
