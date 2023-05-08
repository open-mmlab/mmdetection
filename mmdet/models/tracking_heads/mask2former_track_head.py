# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList
from mmengine.model.weight_init import caffe2_xavier_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads import AnchorFreeHead, MaskFormerHead
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import TrackDataSample, TrackSampleList
from mmdet.structures.mask import mask2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from ..layers import Mask2FormerTransformerDecoder


@MODELS.register_module()
class Mask2FormerTrackHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_classes (int): Number of VIS classes.
        num_queries (int): Number of query in Transformer decoder.
            Defaults to 100.
        num_transformer_feat_level (int): Number of feats levels.
            Defaults to 3.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of transformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding.
            Defaults to `SinePositionalEncoding3D`.
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to 'CrossEntropyLoss'.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to 'DiceLoss'.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_classes: int,
                 num_frames: int = 2,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = MODELS.build(positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            self.sampler = TASK_UTILS.build(
                # self.train_cfg.sampler, default_args=dict(context=self))
                self.train_cfg['sampler'],
                default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(self, batch_gt_instances: InstanceList) -> InstanceList:
        """Preprocess the ground truth for all images.

        It aims to reorganize the `gt`. For example, in the
        `batch_data_sample.gt_instances.mask`, its shape is
        `(all_num_gts, h, w)`, but we don't know each gt belongs to which `img`
        (assume `num_frames` is 2). So, this func used to reshape the `gt_mask`
        to `(num_gts_per_img, num_frames, h, w)`. In addition, we can't
        guarantee that the number of instances in these two images is equal,
        so `-1` refers to nonexistent instances.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``labels``, each is
                ground truth labels of each bbox, with shape (num_gts, )
                and ``masks``, each is ground truth masks of each instances
                of an image, shape (num_gts, h, w).

        Returns:
            list[obj:`InstanceData`]: each contains the following keys

                - labels (Tensor): Ground truth class indices\
                    for an image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in an image.
                - masks (Tensor): Ground truth mask for a\
                    image, with shape (n, t, h, w).
        """
        final_batch_gt_instances = []
        batch_size = len(batch_gt_instances) // self.num_frames
        for batch_idx in range(batch_size):
            pair_gt_insatences = batch_gt_instances[batch_idx *
                                                    self.num_frames:batch_idx *
                                                    self.num_frames +
                                                    self.num_frames]

            assert len(
                pair_gt_insatences
            ) > 1, f'mask2former for vis need multi frames to train, \
                but you only use {len(pair_gt_insatences)} frames'

            _device = pair_gt_insatences[0].labels.device

            for gt_instances in pair_gt_insatences:
                gt_instances.masks = gt_instances.masks.to_tensor(
                    dtype=torch.bool, device=_device)
            all_ins_id = torch.cat([
                gt_instances.instances_ids
                for gt_instances in pair_gt_insatences
            ])
            all_ins_id = all_ins_id.unique().tolist()
            map_ins_id = dict()
            for i, ins_id in enumerate(all_ins_id):
                map_ins_id[ins_id] = i

            num_instances = len(all_ins_id)
            mask_shape = [
                num_instances, self.num_frames,
                pair_gt_insatences[0].masks.shape[1],
                pair_gt_insatences[0].masks.shape[2]
            ]
            gt_masks_per_video = torch.zeros(
                mask_shape, dtype=torch.bool, device=_device)
            gt_ids_per_video = torch.full((num_instances, self.num_frames),
                                          -1,
                                          dtype=torch.long,
                                          device=_device)
            gt_labels_per_video = torch.full((num_instances, ),
                                             -1,
                                             dtype=torch.long,
                                             device=_device)

            for frame_id in range(self.num_frames):
                cur_frame_gts = pair_gt_insatences[frame_id]
                ins_ids = cur_frame_gts.instances_ids.tolist()
                for i, id in enumerate(ins_ids):
                    gt_masks_per_video[map_ins_id[id],
                                       frame_id, :, :] = cur_frame_gts.masks[i]
                    gt_ids_per_video[map_ins_id[id],
                                     frame_id] = cur_frame_gts.instances_ids[i]
                    gt_labels_per_video[
                        map_ins_id[id]] = cur_frame_gts.labels[i]

            tmp_instances = InstanceData(
                labels=gt_labels_per_video,
                masks=gt_masks_per_video.long(),
                instances_id=gt_ids_per_video)
            final_batch_gt_instances.append(tmp_instances)

        return final_batch_gt_instances

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, num_frames, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, num_frames, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        # (num_gts, )
        gt_labels = gt_instances.labels
        # (num_gts, num_frames, h, w)
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)

        # shape (num_queries, num_points)
        mask_points_pred = point_sample(mask_pred,
                                        point_coords.repeat(num_queries, 1,
                                                            1)).flatten(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(gt_masks.float(),
                                       point_coords.repeat(num_gts, 1,
                                                           1)).flatten(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should include
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, num_frames,h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, num_frames, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, num_frames, h, w)
        # -> (num_total_gts, num_frames, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.flatten(0, 1).unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts * num_frames, h, w) ->
            # (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.flatten(0, 1).unsqueeze(1).float(),
                points_coords).squeeze(1)
        # shape (num_total_gts * num_frames, num_points)
        mask_point_preds = point_sample(
            mask_preds.flatten(0, 1).unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_total_gts * num_frames, num_points) ->
        # (num_total_gts * num_frames * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points / self.num_frames)

        return loss_cls, loss_mask, loss_dice

    def _forward_head(
        self, decoder_out: Tensor, mask_feature: Tensor,
        attn_mask_target_size: Tuple[int,
                                     int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, t, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should include background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)

        # shape (batch_size, num_queries, t, h, w)
        mask_pred = torch.einsum('bqc,btchw->bqthw', mask_embed, mask_feature)
        b, q, t, _, _ = mask_pred.shape

        attn_mask = F.interpolate(
            mask_pred.flatten(0, 1),
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False).view(b, q, t, attn_mask_target_size[0],
                                      attn_mask_target_size[1])

        # shape (batch_size, num_queries, t, h, w) ->
        # (batch_size, num_queries, t*h*w) ->
        # (batch_size, num_head, num_queries, t*h*w) ->
        # (batch_size*num_head, num_queries, t*h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(
            self, x: List[Tensor], data_samples: TrackDataSample
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should include background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        bt, c_m, h_m, w_m = mask_features.shape
        batch_size = bt // self.num_frames if self.training else 1
        t = bt // batch_size
        mask_features = mask_features.view(batch_size, t, c_m, h_m, w_m)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2)
            level_embed = self.level_embed.weight[i][None, :, None]
            decoder_input = decoder_input + level_embed
            _, c, hw = decoder_input.shape
            # shape (batch_size*t, c, h, w) ->
            # (batch_size, t, c, hw) ->
            # (batch_size, t*h*w, c)
            decoder_input = decoder_input.view(batch_size, t, c,
                                               hw).permute(0, 1, 3,
                                                           2).flatten(1, 2)
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, t) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                3).permute(0, 1, 3, 2).flatten(1, 2)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def loss(
        self,
        x: Tuple[Tensor],
        data_samples: TrackSampleList,
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the track head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in data_samples:
            video_img_metas = defaultdict(list)
            for image_idx in range(len(data_sample)):
                batch_gt_instances.append(data_sample[image_idx].gt_instances)
                for key, value in data_sample[image_idx].metainfo.items():
                    video_img_metas[key].append(value)
            batch_img_metas.append(video_img_metas)

        # forward
        all_cls_scores, all_mask_preds = self(x, data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances)
        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self,
                x: Tuple[Tensor],
                data_samples: TrackDataSample,
                rescale: bool = True) -> InstanceList:
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            data_samples (List[:obj:`TrackDataSample`]): The Data
                Samples. It usually includes information such as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys
                - labels (Tensor): Prediction class indices\
                    for an image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in an image.
                - masks (Tensor): Prediction mask for a\
                    image, with shape (n, t, h, w).
        """

        batch_img_metas = [
            data_samples[img_idx].metainfo
            for img_idx in range(len(data_samples))
        ]
        all_cls_scores, all_mask_preds = self(x, data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        mask_cls_results = mask_cls_results[0]
        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results[0],
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        results = self.predict_by_feat(mask_cls_results, mask_pred_results,
                                       batch_img_metas)
        return results

    def predict_by_feat(self,
                        mask_cls_results: List[Tensor],
                        mask_pred_results: List[Tensor],
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        """Get top-10 predictions.

        Args:
            mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should include background.
            mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
            batch_img_metas (list[dict]): List of image meta information.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys
                - labels (Tensor): Prediction class indices\
                    for an image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in an image.
                - masks (Tensor): Prediction mask for a\
                    image, with shape (n, t, h, w).
        """
        results = []
        if len(mask_cls_results) > 0:
            scores = F.softmax(mask_cls_results, dim=-1)[:, :-1]
            labels = torch.arange(self.num_classes).unsqueeze(0).repeat(
                self.num_queries, 1).flatten(0, 1).to(scores.device)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(
                10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.num_classes
            mask_pred_results = mask_pred_results[topk_indices]

            img_shape = batch_img_metas[0]['img_shape']
            mask_pred_results = \
                mask_pred_results[:, :, :img_shape[0], :img_shape[1]]
            if rescale:
                # return result in original resolution
                ori_height, ori_width = batch_img_metas[0]['ori_shape'][:2]
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)

            masks = mask_pred_results > 0.

            # format top-10 predictions
            for img_idx in range(len(batch_img_metas)):
                pred_track_instances = InstanceData()

                pred_track_instances.masks = masks[:, img_idx]
                pred_track_instances.bboxes = mask2bbox(masks[:, img_idx])
                pred_track_instances.labels = labels_per_image
                pred_track_instances.scores = scores_per_image
                pred_track_instances.instances_id = torch.arange(10)

                results.append(pred_track_instances)

            return results
