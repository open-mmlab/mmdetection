from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor, nn
from transformers import (AutoTokenizer, BertConfig, BertModel, RobertaConfig,
                          RobertaModel)

from mmdet.models.layers import CdnQueryGenerator
from mmdet.models.layers.transformer.deformable_detr_layers import (
    DeformableDetrTransformerDecoder, DeformableDetrTransformerDecoderLayer)
from mmdet.models.layers.transformer.utils import (MLP, coordinate_to_encoding,
                                                   inverse_sigmoid)
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, OptMultiConfig


@MODELS.register_module()
class UninextTransformerDecoder(DeformableDetrTransformerDecoder):
    """Transformer encoder of DINO."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        # self.norm = nn.LayerNorm(self.embed_dims)
        # in uninext don't use

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
        intermediate_reference_points = []
        # in uninext intermediate_reference_points is 6
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
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + \
                        inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                # intermediate.append(self.norm(query))
                intermediate.append(query)
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class TextCdnQueryGenerator(CdnQueryGenerator):

    def __call__(self, batch_data_samples: SampleList,
                 text_embding: Tensor) -> tuple:
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
        gt_bboxes = torch.cat(gt_bboxes_list)

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat([
            torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)
        ])
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        dn_label_query = text_embding[batch_idx_expand]
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, len(batch_data_samples),
            num_groups)

        attn_mask = self.generate_dn_mask(
            max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            single_paddings=int(max_num_target * 2), dn_num=num_groups)

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta


@MODELS.register_module()
class VL_Align(BaseModule):

    def __init__(self,
                 lan_dims: int = 768,
                 hidden_dims: int = 256,
                 log_scale: float = 0.0,
                 clamp_dot_product: bool = True):
        super().__init__()
        # initialize the bias for focal loss
        bias_init = bias_init_with_prob(0.01)

        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(
            lan_dims, hidden_dims, bias=True)  # 768 -> 256
        self.log_scale = nn.Parameter(
            torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(
            torch.zeros(lan_dims), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(
            torch.Tensor([bias_init]), requires_grad=True)  # size (1,)
        self.clamp_dot_product = clamp_dot_product

    def forward(self, x, embedding):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 768)
        """
        # norm
        embedding = F.normalize(
            embedding, p=2,
            dim=-1)  # (bs, L, 768) L is maximum sentence length
        dot_product_proj_tokens = self.dot_product_projection_text(
            embedding / 2.0)  # 768 -> 256
        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang
        ) + self.bias0  # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        dot_product_proj_queries = self.dot_product_projection_image(
            x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(
            1, A, 1)  # (bs, num_query, L)

        dot_product_logit = (
            torch.matmul(dot_product_proj_queries,
                         dot_product_proj_tokens.transpose(-1, -2)) /
            self.log_scale.exp()
        ) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        if self.clamp_dot_product:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit


@MODELS.register_module()
class DeformableReidHead(BaseModule):

    def __init__(
            self,
            num_layers: int = 2,
            layer_cfg: dict = dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
                cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
            init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self._init_layers()

    def _init_layers(self):

        self.layers = nn.ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.mlp = MLP(self.embed_dims, self.embed_dims, self.embed_dims, 3)

    def forward(self,
                query: Tensor,
                reference_points: Tensor,
                value: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor = None,
                attn_masks: Tensor = None,
                **kwargs) -> Tensor:

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                raise ValueError('reference_points.shape[-1] should be 4')

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            query = layer(
                query,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_masks,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_input,
                **kwargs)

        query = self.mlp(query)

        return query


@MODELS.register_module()
class BertTokenizer(BaseModule):

    def __init__(self,
                 max_length: int = 256,
                 pad_max: bool = True,
                 return_special_tokens_mask: bool = True,
                 return_tensors: str = 'pt',
                 truncation: bool = True) -> None:
        super(BertTokenizer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            'projects/UNINEXT/uninext/bert-base-uncased')
        self.max_length = max_length
        self.pad_max = pad_max
        self.return_special_tokens_mask = return_special_tokens_mask
        self.return_tensors = return_tensors
        self.truncation = truncation

    def forward(self, x, device):
        tokenized = self.tokenizer.batch_encode_plus(
            x,
            max_length=self.max_length,
            padding='max_length' if self.pad_max else 'longest',  # max_length
            return_special_tokens_mask=self.return_special_tokens_mask,
            return_tensors=self.return_tensors,
            truncation=self.truncation).to(device)
        tokenizer_input = {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask
        }
        return tokenizer_input


@MODELS.register_module()
class BertEncoder(BaseModule):

    def __init__(self,
                 bert_name: str = 'bert-base-uncased',
                 use_checkpoint: bool = False,
                 parallel_det: bool = False,
                 output_hidden_states: bool = True,
                 frozen_parameters: bool = True) -> None:
        super(BertEncoder, self).__init__()
        self.bert_name = bert_name

        if self.bert_name == 'bert-base-uncased':
            config = BertConfig.from_pretrained('projects/UNINEXT/uninext/%s' %
                                                self.bert_name)
            config.gradient_checkpointing = use_checkpoint
            self.model = BertModel.from_pretrained(
                'projects/UNINEXT/uninext/%s' % self.bert_name,
                add_pooling_layer=False,
                config=config)
        elif self.bert_name == 'roberta-base':
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = use_checkpoint
            self.model = RobertaModel.from_pretrained(
                self.bert_name, add_pooling_layer=False, config=config)
        else:
            raise NotImplementedError

        self.parallel_det = parallel_det  # False
        self.output_hidden_states = output_hidden_states
        self.frozen_parameters = frozen_parameters

    def forward(self, x, task=None):
        input = x['input_ids']  # (bs, seq_len)
        mask = x['attention_mask']  # (bs, seq_len)

        if self.parallel_det and task == 'detection':
            # disable interaction among tokens
            bs, seq_len = mask.shape
            mask_new = torch.zeros((bs, seq_len, seq_len), device=mask.device)
            for _ in range(bs):
                mask_new[_, :, :] = mask[_]
                num_valid = torch.sum(mask[_])
                mask_new[_, :num_valid, :num_valid] = torch.eye(num_valid)
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask_new,
                output_hidden_states=self.output_hidden_states,
            )
        else:
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=self.output_hidden_states,
            )

        encoded_layers = outputs.hidden_states[1:]

        ret = {'masks': mask, 'hidden': encoded_layers[-1]}
        return ret


@MODELS.register_module()
class ChannelMapperBias(BaseModule):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There would
            be extra_convs when num_outs larger than the length of in_channels.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or dict],
            optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        num_outs: int = None,
        bias: bool = True,  # in uninext use bias in conv
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    bias=bias,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
