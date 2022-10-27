# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from ..layers import MLP, SinePositionalEncoding, inverse_sigmoid
from .deformable_detr import (DeformableDETR, DeformableDetrTransformerDecoder,
                              DeformableDetrTransformerEncoder)


@MODELS.register_module()
class DINO(DeformableDETR):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict): Config of query de-noising.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_query' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `num_query`, and ' \
                '`hidden_dim` are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['num_query'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        self.dn_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize weights for Transformer and other components."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding_cfg)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DinoTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.  # TODO
        self.label_embedding = nn.Embedding(self.bbox_head.cls_out_channels,
                                            self.embed_dims)
        # TODO: Should `label_embedding` be moved to `dn_generator`?

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat). Will only be used when `as_two_stage`
                is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_class`. They are both `None` when 'as_two_stage'
              is `False`.
        """
        dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
            self.dn_generator(batch_data_samples, self.label_embedding)

        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features  # TODO: refine this
        topk = self.num_query  # TODO: two_stage_num_proposals originally
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
        # topk_proposal = torch.gather(
        #     output_proposals, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
        # topk_memory = torch.gather(
        #     output_memory, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, self.embed_dims))
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        # NOTE In the original DeformDETR, init_reference_out is obtained
        # from detached topk_coords_unact, which is different with DINO.  TODO
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:,
                                            None, :].repeat(1, bs,
                                                            1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()

        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_anchor,
            dn_meta=dn_meta)
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self, query: Tensor, memory: Tensor,
                        memory_mask: Tensor, reference_points: Tensor,
                        spatial_shapes: Tensor, level_start_index: Tensor,
                        valid_ratios: Tensor, dn_mask: Tensor) -> Dict:
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_masks=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches)

        inter_states = inter_states.permute(0, 2, 1, 3)

        # if dn_label_query is not None and dn_label_query.size(1) == 0:
        #     # NOTE: If there is no target in the image, the parameters of
        #     # label_embedding won't be used in producing loss, which raises
        #     # RuntimeError when using distributed mode.
        #     inter_states[0] += self.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=references)
        return decoder_outputs_dict


class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self):
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor):
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

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        output = query
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
            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class CdnQueryGenerator:
    # TODO: Should this encapsulation be broken?

    def __init__(self,
                 num_query,
                 hidden_dim,
                 num_classes,
                 noise_scale=dict(label=0.5, box=0.4),
                 group_cfg=dict(
                     dynamic=True, num_groups=None, num_dn_queries=None)):
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

    def __call__(self, batch_data_samples, label_enc=None):
        """
        TODO: Update
        """
        # TODO: temp assert gt_labels is not None and label_enc is not None
        batch_size = len(batch_data_samples)
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
        positive_idx = torch.tensor(range(
            len(boxes))).long().cuda().unsqueeze(0).repeat(num_groups, 1)
        positive_idx += (torch.tensor(range(num_groups)) * len(boxes) *
                         2).long().cuda().unsqueeze(1)
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
            known_bbox_ += \
                torch.mul(rand_part, diff).cuda() * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expand.long().to('cuda')
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand, eps=1e-3)

        padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
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
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
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
