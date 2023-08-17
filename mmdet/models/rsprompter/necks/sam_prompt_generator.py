import copy
import math
from typing import Type, Tuple

import einops
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer
from torch import Tensor

from mmdet.models import SinePositionalEncoding
# from mmpl.registry import MODELS
from mmdet.registry import MODELS
# from mmdet.registry import MODELS
import torch.nn.functional as F


@MODELS.register_module()
class SAMTransformerPromptGenNeck(nn.Module):
    def __init__(
            self,
            prompt_shape=(100, 6),
            in_channels=[1280]*16,
            out_channels=256,
            positional_encoding=dict(num_feats=128, normalize=True),
            n_classes=2,
            kernel_size=3,
            stride=1,
            norm_cfg=None,
            act_cfg=dict(type='ReLU')
    ):
        super(SAMTransformerPromptGenNeck, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_put_channels = out_channels
        self.n_classes = n_classes
        self.stride = stride

        self.prompt_shape = prompt_shape
        self.num_queries = prompt_shape[0]
        self.per_query_point = prompt_shape[1]

        if isinstance(in_channels, list):
            self.pre_layers = nn.ModuleList()
            inner_channel = 32
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
                            inner_channel*2,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            stride=self.stride,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg
                        ),
                        ConvModule(
                            inner_channel*2,
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
                        inner_channel * len(in_channels),
                        out_channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        out_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )

        self.generator_pe = SinePositionalEncoding(**positional_encoding)
        self.transformer = self.build_transformer()
        self.query_feat = nn.Embedding(self.num_queries, out_channels)
        self.query_emb = nn.Embedding(self.num_queries, out_channels)

        self.output_upscaling = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_channels // 4, out_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 8),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_channels // 8, out_channels // 8, kernel_size=3, padding=1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, n_classes)
        )

        # self.point_emb = nn.Sequential(
        #     nn.Linear(out_channels, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, self.per_query_point * out_channels)
        # )
        self.output_hypernetworks_mlps = MLP(out_channels, out_channels, out_channels // 8, 3)


    def build_transformer(
            self, num_encoder_layers=2, num_decoder_layers=3, embed_dims=256, num_heads=8,
            mlp_ratio=2, dropout_rate=0.0, act_cfg=dict(type="gelu")):
        """Build transformer decoder."""
        # transformer = nn.Transformer(
        #     d_model=embed_dims, nhead=num_heads, num_encoder_layers=num_encoder_layers,
        #     num_decoder_layers=num_decoder_layers, dim_feedforward=mlp_ratio * embed_dims,
        #     dropout=dropout_rate, activation=act_cfg['type'], batch_first=True, norm_first=True,
        # )
        transformer = Transformer(depth=2)
        return transformer

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, prompt_encoder, mask_decoder):

        img_embs, inner_states = inputs
        if hasattr(self, 'pre_layers'):
            inner_states = inner_states[-len(self.in_channels):]
            inner_states = [einops.rearrange(x, 'b h w c -> b c h w') for x in inner_states]
            inner_states = [layer(x) for layer, x in zip(self.pre_layers[:-1], inner_states)]
            img_feats = self.pre_layers[-1](torch.cat(inner_states, dim=1))
        bs, c, h, w = img_feats.shape
        mask_pe = torch.zeros((bs, h, w), device=img_feats.device)
        img_feats_pe = self.generator_pe(mask_pe)
        query_feat = self.query_feat.weight.unsqueeze(0).expand(bs, -1, -1)  # Bx256x256
        query_emb = self.query_emb.weight.unsqueeze(0).expand(bs, -1, -1)
        img_feats, query_feats = self.transformer(
            image_embedding=img_feats,
            image_pe=img_feats_pe,
            point_embedding=query_feat,
            point_pe=query_emb)
        cls_logits = self.cls_head(query_feats)
        # point_embs = self.point_emb(query_feats)
        # point_embs = rearrange(point_embs, 'b n (t c) -> b n t c', t=self.per_query_point)  # Bx100x6x256

        src = img_feats.transpose(1, 2).view(bs, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.output_hypernetworks_mlps(query_feats)
        b, c, h, w = upscaled_embedding.shape
        l1_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # dense_masks = einops.rearrange(l1_masks, 'b (n t) h w -> (b n) t h w', t=1)
        # sparse, dense = prompt_encoder(points=None, boxes=None, masks=dense_masks)
        # dense = einops.rearrange(dense, '(b n) t h w -> b n t h w', n=self.num_queries)

        # l2_masks = []
        # iou_preds = []
        # for curr_embedding, sparse_embeddings, dense_embeddings in zip(img_embs, point_embs, dense):
        #     low_res_masks, iou_predictions = mask_decoder(
        #         image_embeddings=curr_embedding.unsqueeze(0),
        #         image_pe=prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings
        #     )
        #     l2_masks.append(low_res_masks[:, 0])
        #     iou_preds.append(iou_predictions[:, 0])
        # l2_masks = torch.stack(l2_masks, dim=0)
        # iou_preds = torch.stack(iou_preds, dim=0)

        l2_masks = None
        iou_preds = None

        return cls_logits, l1_masks, l2_masks, iou_preds


@MODELS.register_module()
class SAMPromptConvNeck(nn.Module):
    def __init__(
            self,
            prompt_shape=(100, 5),
            img_feat_channels=1280,
            out_put_channels=256,
            num_img_feat_level=16,
            n_cls=2,
    ):
        super(SAMPromptConvNeck, self).__init__()
        self.prompt_shape = prompt_shape
        self.num_queries = prompt_shape[0]
        self.per_query_point = prompt_shape[1]
        self.point_size = int(math.sqrt(prompt_shape[0]))

        self.img_feat_channels = img_feat_channels
        self.out_put_channels = out_put_channels
        self.num_img_feat_level = num_img_feat_level
        self.n_cls = n_cls

        # decoder_embed_dims = img_feat_channels // 32
        decoder_embed_dims = 32
        self.decoder_input_projs = nn.ModuleList()
        # from low resolution to high resolution
        for _ in range(num_img_feat_level):
            self.decoder_input_projs.append(
                nn.Sequential(
                    nn.Conv2d(img_feat_channels, decoder_embed_dims, kernel_size=1),
                    # nn.BatchNorm2d(decoder_embed_dims),
                    nn.ReLU(),
                    nn.Conv2d(decoder_embed_dims, decoder_embed_dims, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(decoder_embed_dims),
                    nn.ReLU(),
                ))
        self.level_embed = nn.Embedding(self.num_img_feat_level, decoder_embed_dims)
        self.gather_img_feats = nn.Sequential(
            nn.Conv2d(num_img_feat_level * decoder_embed_dims, out_put_channels, kernel_size=1),
            # nn.BatchNorm2d(out_put_channels),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels * 2, out_put_channels * 2, 3, padding=1),
        )

        self.img_feats_pe = nn.Parameter(torch.zeros(1, out_put_channels*2, self.point_size, self.point_size))

        self.cls_head = nn.Sequential(
            nn.Conv2d(out_put_channels * 2, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, n_cls, 1)
        )

        self.point_emb = nn.Sequential(
            nn.Conv2d(out_put_channels * 2, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, out_put_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_put_channels, self.per_query_point * out_put_channels, 1)
        )

    def forward(self, inputs):
        inner_states = [x.permute(0, 3, 1, 2) for x in inputs]  # from low2high, all 4 layers
        bs = inner_states[0].shape[0]
        # inputs: list([B, C, H, W])
        num_layers = len(inputs)
        # import ipdb; ipdb.set_trace()
        # select the feature maps from the selected layers
        layer_start_id = num_layers - self.num_img_feat_level
        decoder_inputs = []
        for i in range(self.num_img_feat_level):
            decoder_input = self.decoder_input_projs[i](inner_states[i + layer_start_id])  # Bx256x64x64
            level_embed = self.level_embed.weight[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(bs, -1, -1, -1)
            decoder_input = decoder_input + level_embed
            decoder_inputs.append(decoder_input)
        decoder_inputs = torch.cat(decoder_inputs, dim=1)  # Bx256x64x64
        decoder_inputs = self.gather_img_feats(decoder_inputs)
        # import pdb;
        # pdb.set_trace()
        decoder_inputs = torch.nn.functional.interpolate(decoder_inputs, size=(self.point_size, self.point_size), mode='bilinear', align_corners=True)
        img_pe = self.img_feats_pe.expand(bs, -1, -1, -1)  # Bx256x64x64
        decoder_inputs = decoder_inputs + img_pe

        cls_logits = self.cls_head(decoder_inputs)  # b c h w
        cls_logits = rearrange(cls_logits, 'b c h w -> b (h w) c')
        point_embs = self.point_emb(decoder_inputs)  # b c h w
        point_embs = rearrange(point_embs, 'b (t c) h w -> b (h w) t c', t=self.per_query_point)  # Bx100x6x256

        return point_embs, cls_logits




class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        point_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=image_embedding,
                query_pe=image_pe,
                keys=point_embedding,
                key_pe=point_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + image_pe
        k = keys + point_embedding

        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


@MODELS.register_module()
class SAMTransformerEDPromptGenNeck(nn.Module):
    def __init__(
            self,
            prompt_shape=(100, 5),
            in_channels=[1280]*16,
            inner_channels=128,
            selected_channels: list=None,
            num_encoders=2,
            num_decoders=2,
            out_channels=256,
            positional_encoding=dict(num_feats=128, normalize=True),
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            init_cfg=None,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels
        self.stride = stride
        self.selected_channels = selected_channels

        self.prompt_shape = prompt_shape
        self.num_queries = prompt_shape[0]
        self.per_query_point = prompt_shape[1]

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        )

        self.generator_pe = SinePositionalEncoding(**positional_encoding)

        self.en_layers = nn.ModuleList()
        self.de_layers = nn.ModuleList()
        self.build_transformer(num_encoders=num_encoders, num_decoders=num_decoders)

        self.embed_dims = self.en_layers[0].embed_dims
        self.pre_norm = self.en_layers[0].pre_norm

        self.query_feat = nn.Embedding(self.num_queries, out_channels)
        self.query_embed = nn.Embedding(self.num_queries, out_channels)

        # self.output_upscaling = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.GELU(),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels // 4),
        #     nn.GELU(),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(out_channels // 4, out_channels // 8, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels // 8),
        #     nn.GELU(),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(out_channels // 8, out_channels // 8, kernel_size=3, padding=1),
        # )
        # self.output_hypernetworks_mlps = MLP(out_channels, out_channels, out_channels // 8, 3)

        self.init_weights()

    def build_transformer(self, num_encoders=2, num_decoders=2, embed_dims=256, num_heads=8, mlp_ratio=4):
        transformer_encoder_layer = dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=dict(type='Dropout', drop_prob=0.1)
                ),
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * mlp_ratio,
                num_fcs=2,
                act_cfg=dict(type='GELU'),
                ffn_drop=0.1,
                add_identity=True),
            operation_order=('norm', 'self_attn', 'norm', 'ffn'),
            norm_cfg=dict(type='LN'),
            batch_first=True
        )
        transformer_decoder_layer = dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=dict(type='Dropout', drop_prob=0.1)
                ),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * mlp_ratio,
                num_fcs=2,
                act_cfg=dict(type='GELU'),
                ffn_drop=0.1,
                add_identity=True),
            operation_order=('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn'),
            norm_cfg=dict(type='LN'),
            batch_first=True
        )

        transformer_en_layers = [
            copy.deepcopy(transformer_encoder_layer) for _ in range(num_encoders)
        ]
        transformer_de_layers = [
            copy.deepcopy(transformer_decoder_layer) for _ in range(num_decoders)
        ]
        for i in range(num_encoders):
            self.en_layers.append(build_transformer_layer(transformer_en_layers[i]))
        for i in range(num_decoders):
            self.de_layers.append(build_transformer_layer(transformer_de_layers[i]))

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs):
        _, inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats = self.up_layers[1](x)

        bs, c, h, w = img_feats.shape

        mask_pe = torch.zeros((bs, h, w), device=img_feats.device, dtype=torch.bool)
        img_feats_pe = self.generator_pe(mask_pe)

        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (bs, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (bs, 1, 1))

        encoder_inputs = rearrange(img_feats, 'b c h w -> b (h w) c')
        img_feats_pe = img_feats_pe.flatten(2).permute(0, 2, 1)

        # shape (batch_size, num_total_queries, c)
        memory = encoder_inputs
        for layer in self.en_layers:
            memory = layer(
                query=memory,
                query_pos=img_feats_pe
            )
        # (batch_size, num_total_queries, c)

        query_feat_list = []
        for layer in self.de_layers:
            query_feat = layer(
                query=query_feat,
                key=memory,
                value=memory,
                query_pos=query_embed,
                key_pos=img_feats_pe
            )
            query_feat_list.append(query_feat)

        img_feat = rearrange(memory, 'b (h w) c -> b c h w', h=h, w=w)
        return query_feat, query_feat_list, img_feat


@MODELS.register_module()
class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            in_channels=[1280]*16,
            inner_channels=128,
            selected_channels: list=None,
            out_channels=256,
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
            up_sample_scale=4,
            init_cfg=None,
            **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_channels = out_channels
        self.stride = stride
        self.selected_channels = selected_channels
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=None
            )
        )

        self.up_sample_layers = nn.ModuleList()
        assert up_sample_scale == 4
        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, inputs):
        _, inner_states = inputs
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0