# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import Linear, build_norm_layer
from mmcv.cnn.bricks.drop import Dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, ModuleList, uniform_init, xavier_init
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import (MLP, DetrTransformerDecoder, DetrTransformerDecoderLayer,
                      DetrTransformerEncoder, DetrTransformerEncoderLayer,
                      SinePositionalEncodingHW)
from ..layers.transformer import gen_sineembed_for_position, inverse_sigmoid
from .base_detr import TransformerDetector


@MODELS.register_module()
class DABDETR(TransformerDetector):

    def __init__(self,
                 *args,
                 iter_update=True,
                 random_refpoints_xy=False,
                 num_patterns=0,
                 **kwargs) -> None:
        self.iter_update = iter_update
        self.random_refpoints_xy = random_refpoints_xy
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning('num_patterns should be int but {}'.format(
                type(num_patterns)))
            self.num_patterns = 0

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncodingHW(
            **self.positional_encoding_cfg)
        self.encoder = DabDetrTransformerEncoder(**self.encoder_cfg)
        self.decoder = DabDetrTransformerDecoder(**self.decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.query_embedding = nn.Embedding(self.num_query, self.query_dim)
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        # self.bbox_embed_diff_each_layer = \
        #     self.decoder.bbox_embed_diff_each_layer
        self.nb_dec = self.decoder.num_layers

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

    def init_weights(self) -> None:
        super(TransformerDetector, self).init_weights()
        self._init_transformer_weights()
        if self.random_refpoints_xy:
            uniform_init(self.query_embedding)
            self.query_embedding.weight.data[:, :2] = \
                inverse_sigmoid(self.query_embedding.weight.data[:, :2])
            self.query_embedding.weight.data[:, :2].requires_grad = False

    def _init_transformer_weights(self) -> None:
        # follow the DetrTransformer to init parameters
        for coder in [self.encoder, self.decoder]:
            for m in coder.modules():
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')

    def forward_pretransformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Dict[str, Tensor]:
        feat = img_feats[-1]
        batch_size = feat.size(0)
        # construct binary masks which used for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [
            sample.img_shape  # noqa
            for sample in batch_data_samples
        ]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.

        # prepare transformer_inputs_dict
        masks = F.interpolate(
            masks.unsqueeze(1), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # iterative refinement for anchor boxes
        reg_branches = self.bbox_head.fc_reg if self.iter_update else None

        transformer_inputs_dict = dict(
            feat=feat,
            masks=masks,
            pos_embed=pos_embed,
            query_embed=self.query_embedding.weight,
            reg_branches=reg_branches)
        return transformer_inputs_dict  # noqa

    def forward_transformer(self,
                            feat: Tensor,
                            masks: Tensor,
                            pos_embed: Tensor,
                            query_embed: nn.Module,
                            return_memory: bool = False,
                            reg_branches=None) -> Union[Tuple[Tensor], Any]:
        bs, c, h, w = feat.shape
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        feat = feat.view(bs, c, -1).permute(2, 0,
                                            1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, 4] -> [num_query, bs, 4]
        masks = masks.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=feat, query_pos=pos_embed, query_key_padding_mask=masks)
        if self.num_patterns == 0:
            target = torch.zeros(
                self.num_query, bs, self.embed_dims, device=query_embed.device)
        else:
            target = self.patterns.weight[:, None, None, :]\
                .repeat(1, self.num_query, bs, 1)\
                .view(-1, bs, self.embed_dims)
            query_embed = query_embed.repeat(self.num_patterns, 1, 1)

        out_dec, reference = self.decoder(
            query=target,
            key=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=masks,
            reg_branches=reg_branches)
        if return_memory:
            memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
            return out_dec, reference, memory
        return out_dec, reference, None

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Tuple[Tensor]:
        """Network forward process.

        Includes backbone, neck and head forward without post-processing.
        """
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec, reference, _ = self.forward_transformer(
            **transformer_inputs_dict)
        results = self.bbox_head.forward(outs_dec, reference)
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec, reference, _ = self.forward_transformer(
            **transformer_inputs_dict)
        losses = self.bbox_head.loss(outs_dec, reference, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        transformer_inputs_dict = self.forward_pretransformer(
            img_feats, batch_data_samples)
        outs_dec, reference, _ = self.forward_transformer(
            **transformer_inputs_dict)
        results_list = self.bbox_head.predict(
            outs_dec, reference, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


class ConditionalAttention(BaseModule):
    """A wrapper of conditional attention, dropout and residual connection."""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_dropout=0.,
                 proj_drop=0.,
                 batch_first=False,
                 cross_attn=False,
                 keep_query_pos=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.batch_first = batch_first  # indispensable
        self.cross_attn = cross_attn
        self.keep_query_pos = keep_query_pos
        self.embed_dims = embed_dims  # output dims
        self.num_heads = num_heads
        self.attn_dropout = Dropout(attn_dropout)
        self.proj_drop = Dropout(proj_drop)

        self._init_proj()

    def _init_proj(self):
        embed_dims = self.embed_dims
        self.qcontent_proj = Linear(embed_dims, embed_dims)
        self.qpos_proj = Linear(embed_dims, embed_dims)
        self.kcontent_proj = Linear(embed_dims, embed_dims)
        self.kpos_proj = Linear(embed_dims, embed_dims)
        self.v_proj = Linear(embed_dims, embed_dims)
        if self.cross_attn:
            self.qpos_sine_proj = Linear(embed_dims, embed_dims)
        self.out_proj = Linear(embed_dims, embed_dims)

        nn.init.constant_(self.out_proj.bias, 0.)  # init out_proj

    def forward_attn(self, query, key, value, attn_mask, key_padding_mask,
                     need_weights):
        assert key.size(0) == value.size(0), \
            f'{"key, value must have the same sequence length"}'
        assert query.size(1) == key.size(1) == value.size(1), \
            f'{"batch size must be equal for query, key, value"}'
        assert query.size(2) == key.size(2), \
            f'{"q_dims, k_dims must be equal"}'
        assert value.size(2) == self.embed_dims, \
            f'{"v_dims must be equal to embed_dims"}'

        tgt_len, bsz, hidden_dims = query.size()
        head_dims = hidden_dims // self.num_heads
        v_head_dims = self.embed_dims // self.num_heads
        assert head_dims * self.num_heads == hidden_dims, \
            f'{"hidden_dims must be divisible by num_heads"}'
        scaling = float(head_dims)**-0.5

        q = query * scaling
        k = key
        v = value

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or \
                   attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or \
                   attn_mask.dtype == torch.uint8 or \
                   attn_mask.dtype == torch.bool, \
                   'Only float, byte, and bool types are supported for \
                    attn_mask'

            if attn_mask.dtype == torch.uint8:
                warnings.warn('Byte tensor for attn_mask is deprecated.\
                     Use bool tensor instead.')
                attn_mask = attn_mask.to(torch.bool)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError(
                        'The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                        bsz * self.num_heads,
                        query.size(0),
                        key.size(0)
                ]:
                    raise RuntimeError(
                        'The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(
                        attn_mask.dim()))
        # attn_mask's dim is 3 now.

        if key_padding_mask is not None and key_padding_mask.dtype == int:
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                head_dims).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads,
                                    head_dims).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads,
                                    v_head_dims).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads, tgt_len, src_len
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(
            attn_output_weights -
            attn_output_weights.max(dim=-1, keepdim=True)[0],
            dim=-1)
        attn_output_weights = self.attn_dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [
            bsz * self.num_heads, tgt_len, v_head_dims
        ]
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bsz, self.embed_dims)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

    def forward(
            self,
            query,  # tgt
            key,  # memory
            query_pos=None,
            query_sine_embed=None,
            key_pos=None,  # pos
            attn_mask=None,
            query_key_padding_mask=None,  # tgt_key_padding_mask
            key_padding_mask=None,  # memory_key_padding_mask
            need_weights=True,
            is_first=False):
        if self.cross_attn:
            q_content = self.qcontent_proj(query)
            k_content = self.kcontent_proj(key)
            v = self.v_proj(key)

            nq, bs, c = q_content.size()
            hw, _, _ = k_content.size()

            k_pos = self.kpos_proj(key_pos)
            if is_first or self.keep_query_pos:
                q_pos = self.qpos_proj(query_pos)
                q = q_content + q_pos
                k = k_content + k_pos
            else:
                q = q_content
                k = k_content
            q = q.view(nq, bs, self.num_heads, c // self.num_heads)
            query_sine_embed = self.qpos_sine_proj(query_sine_embed)
            query_sine_embed = query_sine_embed.view(nq, bs, self.num_heads,
                                                     c // self.num_heads)
            q = torch.cat([q, query_sine_embed], dim=3).view(nq, bs, 2 * c)
            k = k.view(hw, bs, self.num_heads, c // self.num_heads)
            k_pos = k_pos.view(hw, bs, self.num_heads, c // self.num_heads)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, 2 * c)
            ca_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights)[0]
            query = query + self.proj_drop(ca_output)
        else:
            q_content = self.qcontent_proj(query)
            q_pos = self.qpos_proj(query_pos)
            k_content = self.kcontent_proj(query)
            k_pos = self.kpos_proj(query_pos)
            v = self.v_proj(query)
            q = q_content if q_pos is None else q_content + q_pos
            k = k_content if k_pos is None else k_content + k_pos
            sa_output = self.forward_attn(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                key_padding_mask=query_key_padding_mask,
                need_weights=need_weights)[0]
            query = query + self.proj_drop(sa_output)
        return query


class DabDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        self.self_attn = ConditionalAttention(**self.self_attn_cfg)
        self.cross_attn = ConditionalAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.keep_query_pos = self.cross_attn.keep_query_pos

    def forward(
            self,
            query,  # tgt
            key=None,  # memory
            query_pos=None,
            query_sine_embed=None,
            key_pos=None,  # pos
            self_attn_mask=None,
            cross_attn_mask=None,
            query_key_padding_mask=None,  # tgt_key_padding_mask
            key_padding_mask=None,  # memory_key_padding_mask
            need_weights=True,
            is_first=False,
            **kwargs):

        query = self.self_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            key_pos=key_pos,
            attn_mask=self_attn_mask,
            key_padding_mask=key_padding_mask,
            query_key_padding_mask=query_key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            query_key_padding_mask=query_key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class DabDetrTransformerDecoder(DetrTransformerDecoder):

    def __init__(
            self,
            *args,
            query_dim=4,
            query_scale_type='cond_elewise',
            modulate_hw_attn=True,
            # bbox_embed_diff_each_layer=False,
            **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.modulate_hw_attn = modulate_hw_attn
        # self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        assert self.query_dim in [2, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                DabDetrTransformerDecoderLayer(**self.layer_cfg[i]))

        embed_dims = self.layers[0].embed_dims
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP(self.query_dim // 2 * embed_dims, embed_dims,
                                  embed_dims, 2)

        if self.modulate_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None

    def forward(
            self,
            query,  # tgt
            key,  # memory
            query_pos,  # refpoints_unsigmoid
            key_pos=None,  # pos
            attn_masks=None,
            query_key_padding_mask=None,  # tgt_key_padding_mask
            key_padding_mask=None,  # memory_key_padding_mask
            reg_branches=None,
            **kwargs):

        output = query
        reference_unsigmoid = query_pos

        reference = reference_unsigmoid.sigmoid()
        ref = [reference]

        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = reference[..., :self.query_dim]
            query_sine_embed = gen_sineembed_for_position(
                pos_tensor=obj_center, num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(
                query_sine_embed)  # [nq, bs, 2c] -> [nq, bs, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            query_sine_embed = query_sine_embed[
                ..., :self.embed_dims] * pos_transformation
            # modulated height and weight attention
            if self.modulate_hw_attn:
                assert obj_center.size(-1) == 4
                ref_hw = self.ref_anchor_head(output).sigmoid()
                query_sine_embed[..., self.embed_dims // 2:] *= \
                    (ref_hw[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., : self.embed_dims // 2] *= \
                    (ref_hw[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = layer(
                output,
                key,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                key_pos=key_pos,
                self_attn_mask=attn_masks,
                cross_attn_mask=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            # iter update
            if reg_branches is not None:
                # if self.bbox_embed_diff_each_layer:
                #     tmp = reg_branches[layer_id](output)
                # else:
                #     tmp = reg_branches(output)
                tmp = reg_branches(output)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference)
                new_reference = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref.append(new_reference)
                reference = new_reference.detach()  # no grad_fn
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(output))
                else:
                    intermediate.append(output)

        if self.post_norm is not None:
            output = self.post_norm(output)

        if reg_branches is not None and self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(1, 2),
                torch.stack(ref).transpose(1, 2),
            ]
        elif reg_branches is None and self.return_intermediate:
            return [
                torch.stack(intermediate).transpose(
                    1, 2),  # return_intermediate is True
                reference.unsqueeze(0).transpose(1, 2)  # reg_branches is None
            ]
        elif reg_branches is None and not self.return_intermediate:
            return [
                output.unsqueeze(0).transpose(
                    1, 2),  # return_intermediate is False
                reference.unsqueeze(0).transpose(1, 2)  # reg_branches is None
            ]
        else:
            return [
                output.unsqueeze(0).transpose(
                    1, 2),  # return_intermediate is False
                torch.stack(ref).transpose(1, 2)  # reg_branches is not None
            ]


class DabDetrTransformerEncoder(DetrTransformerEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        self.layers = ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                DetrTransformerEncoderLayer(**self.layer_cfg[i]))
        embed_dims = self.layers[0].embed_dims
        self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)

    def forward(self,
                query,
                query_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                **kwargs):
        for layer in self.layers:
            pos_scales = self.query_scale(query)
            query = layer(
                query,
                query_pos=query_pos * pos_scales,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                **kwargs)
        return query
