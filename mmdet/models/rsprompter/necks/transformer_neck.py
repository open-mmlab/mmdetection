import copy

import torch
import torch.nn as nn
# from mmpl.registry import MODELS
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer


@MODELS.register_module()
class TransformerEncoderNeck(BaseModule):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self,
                 model_dim,
                 with_pe=True,
                 max_position_embeddings=24,
                 with_cls_token=True,
                 num_encoder_layers=3
                 ):
        super(TransformerEncoderNeck, self).__init__()
        self.embed_dims = model_dim
        self.with_cls_token = with_cls_token
        self.with_pe = with_pe

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        if self.with_pe:
            self.pe = nn.Parameter(torch.zeros(1, max_position_embeddings, self.embed_dims))

        mlp_ratio = 4
        embed_dims = model_dim
        transformer_layer = dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=8,
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

        self.layers = nn.ModuleList()
        transformer_layers = [
            copy.deepcopy(transformer_layer) for _ in range(num_encoder_layers)
        ]
        for i in range(num_encoder_layers):
            self.layers.append(build_transformer_layer(transformer_layers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B = x.shape[0]
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.with_pe:
            x = x + self.pe[:, :x.shape[1], :]
        for layer in self.layers:
            x = layer(x)

        if self.with_cls_token:
            return x[:, 0], x
        return None, x
