# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from .deformable_detr_transformer import DeformableDetrTransformerDecoder
from .utils import MLP, inverse_sigmoid


class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer encoder of DINO."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)  # TODO: refine this

    @staticmethod
    def gen_sineembed_for_position(pos_tensor: Tensor):  # TODO: rename this
        # TODO: Qizhi and Yiming seem add this function in utils.py
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            128, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000**(2 * (dim_t // 2) / 128)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(
            self,
            query: Tensor,
            value: Tensor,
            key_padding_mask: Tensor,
            self_attn_mask: Tensor,
            reference_points: Tensor,
            spatial_shapes: Tensor,
            level_start_index: Tensor,
            valid_ratios: Tensor,
            reg_branches: nn.
        ModuleList,  # TODO: why not ModuleList in mmcv?  # noqa):
            **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (num_query, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_query, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_query_total, num_query_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_query, 4).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (num_query, bs, dim)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
            query = query.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier?
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            query = query.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class CdnQueryGenerator(BaseModule):
    # TODO: Should this encapsulation be broken?

    def __init__(self,
                 num_query,
                 hidden_dim,
                 num_classes,
                 noise_scale=dict(label=0.5, box=0.4),
                 group_cfg=dict(
                     dynamic=True, num_groups=None, num_dn_queries=None)):
        super().__init__()
        self.num_query = num_query
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.  # TODO
        self.label_embedding = nn.Embedding(self.num_classes + 1,
                                            self.hidden_dim)
        # TODO: Be careful with the init of the label_embedding

    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def __call__(self, batch_data_samples):
        """
        TODO: Update
        """
        # TODO: add assert gt_labels is not None
        batch_size = len(batch_data_samples)
        device = batch_data_samples[0].gt_instances.bboxes.device
        # convert bbox
        gt_labels = []
        gt_bboxes = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes) / factor
            gt_bboxes.append(bboxes_normalized)
            gt_labels.append(sample.gt_instances.labels)

        known = [torch.ones_like(labels) for labels in gt_labels]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num)))

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes)
        batch_idx = torch.cat(
            [torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * num_groups, 1).view(-1)
        known_labels = labels.repeat(2 * num_groups, 1).view(-1)
        known_bid = batch_idx.repeat(2 * num_groups, 1).view(-1)
        known_bboxs = boxes.repeat(2 * num_groups, 1)
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expand.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))  # TODO

        pad_size = int(single_pad * 2 * num_groups)
        positive_idx = torch.arange(
            len(bboxes), dtype=torch.long,
            device=device)  # TODO: replace the `len(bboxes)`  # noqa
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32)
            rand_sign = rand_sign * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ += torch.mul(rand_part,
                                     diff).to(device) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expand.long().to(device)
        input_label_embed = self.label_embedding(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand, eps=1e-3)

        padding_label = torch.zeros(pad_size, self.hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(2 * num_groups)
            ]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(),
                               map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(),
                              map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + self.num_query
        attn_mask = torch.ones(tgt_size, tgt_size, device=device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
            if i == num_groups - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1),
                          single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 *
                          (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': num_groups,
        }
        return input_query_label, input_query_bbox, attn_mask, dn_meta
