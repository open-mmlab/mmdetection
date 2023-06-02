# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, Union

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import OptConfigType
from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid


class DinoTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer encoder of DINO."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
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
            (num_queries, bs, dim)
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

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

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
    """Implement query generator of the Contrastive denoising (CDN) proposed in
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
    Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        num_classes (int): Number of object classes.
        embed_dims (int): The embedding dimensions of the generated queries.
        num_matching_queries (int): The queries number of the matching part.
            Used for generating dn_mask.
        label_noise_scale (float): The scale of label noise, defaults to 0.5.
        box_noise_scale (float): The scale of box noise, defaults to 1.0.
        group_cfg (:obj:`ConfigDict` or dict, optional): The config of the
            denoising queries grouping, includes `dynamic`, `num_dn_queries`,
            and `num_groups`. Two grouping strategies, 'static dn groups' and
            'dynamic dn groups', are supported. When `dynamic` is `False`,
            the `num_groups` should be set, and the number of denoising query
            groups will always be `num_groups`. When `dynamic` is `True`, the
            `num_dn_queries` should be set, and the group number will be
            dynamic to ensure that the denoising queries number will not exceed
            `num_dn_queries` to prevent large fluctuations of memory. Defaults
            to `None`.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            if 'num_dn_queries' not in group_cfg:
                warnings.warn("'num_dn_queries' should be set when using "
                              'dynamic dn groups, use 100 as default.')
            self.num_dn_queries = group_cfg.get('num_dn_queries', 100)
            assert isinstance(self.num_dn_queries, int), \
                f'Expected the num_dn_queries to have type int, but got ' \
                f'{self.num_dn_queries}({type(self.num_dn_queries)}). '
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), \
                f'Expected the num_groups to have type int, but got ' \
                f'{self.num_groups}({type(self.num_groups)}). '

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.
        # TODO: num_classes + 1 or num_classes ?
        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)

    def __call__(self, batch_data_samples: SampleList) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth.

        Descriptions of the Number Values in code and comments:
            - num_target_total: the total target number of the input batch
              samples.
            - max_num_target: the max target number of the input batch samples.
            - num_noisy_targets: the total targets number after adding noise,
              i.e., num_target_total * num_groups * 2.
            - num_denoising_queries: the length of the output batched queries,
              i.e., max_num_target * num_groups * 2.

        NOTE The format of input bboxes in batch_data_samples is unnormalized
        (x, y, x, y), and the output bbox queries are embedded by normalized
        (cx, cy, w, h) format bboxes going through inverse_sigmoid.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): List of the batch
                data samples, each includes `gt_instance` which has attributes
                `bboxes` and `labels`. The `bboxes` has unnormalized coordinate
                format (x, y, x, y).

        Returns:
            tuple: The outputs of the dn query generator.

            - dn_label_query (Tensor): The output content queries for denoising
              part, has shape (bs, num_denoising_queries, dim), where
              `num_denoising_queries = max_num_target * num_groups * 2`.
            - dn_bbox_query (Tensor): The output reference bboxes as positions
              of queries for denoising part, which are embedded by normalized
              (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
              shape (bs, num_denoising_queries, 4) with the last dimension
              arranged as (cx, cy, w, h).
            - attn_mask (Tensor): The attention mask to prevent information
              leakage from different denoising groups and matching parts,
              will be used as `self_attn_mask` of the `decoder`, has shape
              (num_queries_total, num_queries_total), where `num_queries_total`
              is the sum of `num_denoising_queries` and `num_matching_queries`.
            - dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
        """
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int = None) -> int:
        """Calculate denoising query groups number.

        Two grouping strategies, 'static dn groups' and 'dynamic dn groups',
        are supported. When `self.dynamic_dn_groups` is `False`, the number
        of denoising query groups will always be `self.num_groups`. When
        `self.dynamic_dn_groups` is `True`, the group number will be dynamic,
        ensuring the denoising queries number will not exceed
        `self.num_dn_queries` to prevent large fluctuations of memory.

        NOTE The `num_group` is shared for different samples in a batch. When
        the target numbers in the samples varies, the denoising queries of the
        samples containing fewer targets are padded to the max length.

        Args:
            max_num_target (int, optional): The max target number of the batch
                samples. It will only be used when `self.dynamic_dn_groups` is
                `True`. Defaults to `None`.

        Returns:
            int: The denoising group number of the current batch.
        """
        if self.dynamic_dn_groups:
            assert max_num_target is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(self, gt_labels: Tensor,
                                num_groups: int) -> Tensor:
        """Generate noisy labels and their query embeddings.

        The strategy for generating noisy labels is: Randomly choose labels of
        `self.label_noise_scale * 0.5` proportion and override each of them
        with a random object category label.

        NOTE Not add noise to all labels. Besides, the `self.label_noise_scale
        * 0.5` arg is the ratio of the chosen positions, which is higher than
        the actual proportion of noisy labels, because the labels to override
        may be correct. And the gap becomes larger as the number of target
        categories decreases. The users should notice this and modify the scale
        arg or the corresponding logic according to specific dataset.

        Args:
            gt_labels (Tensor): The concatenated gt labels of all samples
                in the batch, has shape (num_target_total, ) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The query embeddings of noisy labels, has shape
            (num_noisy_targets, embed_dims), where `num_noisy_targets =
            num_target_total * num_groups * 2`.
        """
        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups,
                                            1).view(-1)  # Note `* 2`  # noqa
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(
            -1)  # Note `* 0.5`
        new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice,
                                                       new_labels)
        dn_label_query = self.label_embedding(noisy_labels_expand)
        return dn_label_query

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(
            len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(
            num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(
            rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return dn_bbox_query

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, batch_idx: Tensor,
                           batch_size: int, num_groups: int) -> Tuple[Tensor]:
        """Collate generated queries to obtain batched dn queries.

        The strategy for query collation is as follow:

        .. code:: text

                    input_queries (num_target_total, query_dim)
            P_A1 P_B1 P_B2 N_A1 N_B1 N_B2 P'A1 P'B1 P'B2 N'A1 N'B1 N'B2
              |________ group1 ________|    |________ group2 ________|
                                         |
                                         V
                      P_A1 Pad0 N_A1 Pad0 P'A1 Pad0 N'A1 Pad0
                      P_B1 P_B2 N_B1 N_B2 P'B1 P'B2 N'B1 N'B2
                       |____ group1 ____| |____ group2 ____|
             batched_queries (batch_size, max_num_target, query_dim)

            where query_dim is 4 for bbox and self.embed_dims for label.
            Notation: _-group 1; '-group 2;
                      A-Sample1(has 1 target); B-sample2(has 2 targets)

        Args:
            input_label_query (Tensor): The generated label queries of all
                targets, has shape (num_target_total, embed_dims) where
                `num_target_total = sum(num_target_list)`.
            input_bbox_query (Tensor): The generated bbox queries of all
                targets, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_idx (Tensor): The batch index of the corresponding sample
                for each target, has shape (num_target_total).
            batch_size (int): The size of the input batch.
            num_groups (int): The number of denoising query groups.

        Returns:
            tuple[Tensor]: Output batched label and bbox queries.
            - batched_label_query (Tensor): The output batched label queries,
              has shape (batch_size, max_num_target, embed_dims).
            - batched_bbox_query (Tensor): The output batched bbox queries,
              has shape (batch_size, max_num_target, 4) with the last dimension
              arranged as (cx, cy, w, h).
        """
        device = input_label_query.device
        num_target_list = [
            torch.sum(batch_idx == idx) for idx in range(batch_size)
        ]
        max_num_target = max(num_target_list)
        num_denoising_queries = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([
            torch.arange(num_target, device=device)
            for num_target in num_target_list
        ])
        map_query_index = torch.cat([
            map_query_index + max_num_target * i for i in range(2 * num_groups)
        ]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(
            batch_size, num_denoising_queries, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(
            batch_size, num_denoising_queries, 4, device=device)

        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.

        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask
