import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER


@TRANSFORMER_LAYER.register_module()
class DetrTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs.
        norm_cfg (dict): Config dict for normalization layer.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            operation_order=operation_order,
            ffn_dropout=ffn_dropout,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(self.operation_order) == 4
        assert set(self.operation_order) == set(['self_attn', 'norm', 'ffn'])


@TRANSFORMER_LAYER.register_module()
class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(
            self,
            *args,
            coder_norm_cfg=dict(type='LN'),
            **kwargs,
    ):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(
                coder_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify coder_norm_cfg'
            self.coder_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.coder_norm is not None:
            x = self.coder_norm(x)
        return x

class DeformableDetrTransformerDecoder(BaseTransformerCoder):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):

        super(DeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        # TODO make this when as two stage
        self.bbox_embed = None

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            # hack implementation for iterative bounding box refinement
            def inverse_sigmoid(x, eps=1e-5):
                x = x.clamp(min=0, max=1)
                x1 = x.clamp(min=eps)
                x2 = (1 - x).clamp(min=eps)
                return torch.log(x1 / x2)

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
            # TODO remove this
            output = output.permute(1, 0, 2)

            if self.return_intermediate:

                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 coder_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(DetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if coder_norm_cfg is not None:
            self.coder_norm = build_norm_layer(coder_norm_cfg,
                                               self.embed_dims)[1]
        else:
            self.coder_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.coder_norm:
                x = self.coder_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.coder_norm is not None:
                    intermediate.append(self.coder_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@TRANSFORMER.register_module()
class Transformer(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 encoder=dict(
                     transformerlayers=None,
                     num_encoder_layers=6,
                     coder_norm_cfg=None,
                 ),
                 decoder=dict(
                     transformerlayers=None,
                     num_decoder_layers=6,
                     coder_norm_cfg=None,
                 ),
                 init_cfg=None):
        super(Transformer, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer`.

        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.flatten(1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory


@TRANSFORMER.register_module()
class DeformableDetrTransformer(nn.Module):
    """Implements the Deformable Detr  transformer.

    Args:
        attn_cfgs (sequence[obj:`mmcv.Config`]) : Config of
            Attention in Encoder and decoder in Transformer.
        num_encoder_layers (int): Number of `TransformerEncoderLayer`.
        num_decoder_layers (int): Number of `TransformerDecoderLayer`.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4
        feedforward_channels (int): The hidden dimension for FFNs used in both
            encoder and decoder.
        ffn_dropout (float): Probability of an element to
            be zeroed. Default 0.0.
        act_cfg (dict): Activation config for FFNs used in both encoder
            and decoder. Defalut: ReLU.
        norm_cfg (dict): Config dict for normalization used in both encoder
            and decoder. Default layer normalization.
        num_fcs (int): The number of fully-connected layers in FFNs, which is
            used for both encoder and decoder.
        pre_norm (bool): Whether the normalization layer is ordered
            first in the encoder and decoder. Default False.
        return_intermediate_dec (bool): Whether to return the intermediate
            output from each TransformerDecoderLayer or only the last
            TransformerDecoderLayer. Default False. If False, the returned
            `hs` has shape [num_decoder_layers, bs, num_query, embed_dims].
            If True, the returned `hs` will have shape [1, bs, num_query,
            embed_dims].
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 encodelayers=None,
                 dencodelayers=None,
                 embed_dims=256,
                 as_two_stage=False,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_feature_levels=4,
                 norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 two_stage_num_proposals=300):
        super(DeformableDetrTransformer, self).__init__()
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        self.embed_dims = embed_dims
        self.two_stage_num_proposals = two_stage_num_proposals

        encodelayers = copy.deepcopy(encodelayers)
        dncodelayers = copy.deepcopy(dencodelayers)
        if isinstance(encodelayers, ConfigDict):
            encodelayers = [
                copy.deepcopy(encodelayers) for _ in range(num_encoder_layers)
            ]
        if isinstance(dencodelayers, ConfigDict):
            dencodelayers = [
                copy.deepcopy(dncodelayers) for _ in range(num_encoder_layers)
            ]
        self.encoder = DetrTransformerEncoder(
            transformerlayers=encodelayers,
            num_layers=num_encoder_layers,
            coder_norm_cfg=norm_cfg,
        )
        self.decoder = DeformableDetrTransformerDecoder(
            transformerlayers=dencodelayers,
            num_layers=num_decoder_layers,
            return_intermediate=return_intermediate)

        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution=distribution)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        assert self.as_two_stage or query_embed is not None
        # TODO: make all to (num_query, bs, num_dims)
        # prepare input for encoder
        # TODO : move these to encoder may be better ?
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        # encoder
        # TODO change the dim order
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        # prepare input for decoder
        # move these to decoder may be batter ?
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[
                self.decoder.num_layers](
                    output_memory)
            enc_outputs_coord_unact = \
                self.decoder.bbox_embed[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embeds, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embeds, tgt = torch.split(query_embed, c, dim=1)
            query_embeds = query_embeds.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embeds).sigmoid()
            init_reference_out = reference_points

        # decoder
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_embeds = query_embeds.permute(1, 0, 2)
        hs, inter_references = self.decoder(
            query=tgt,
            key=None,
            value=memory,
            query_pos=query_embeds,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        inter_references_out = inter_references
        if self.as_two_stage:
            return hs, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        return hs, init_reference_out, \
            inter_references_out, None, None


@TRANSFORMER.register_module()
class DynamicConv(BaseModule):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        act_cfg (dict): The activation config for DynamicConv.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(DynamicConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        self.fc_layer = nn.Linear(num_output, self.out_channels)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        num_proposals = param_feature.size(0)
        input_feature = input_feature.view(num_proposals, self.in_channels,
                                           -1).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features
