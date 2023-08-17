import torch
import torch.nn as nn

# from mmpl.registry import MODELS
from mmdet.registry import MODELS


@MODELS.register_module()
class TransformerEDecoderNeck(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, model_dim, num_encoder_layers=3):
        super(TransformerEDecoderNeck, self).__init__()
        self.embed_dims = model_dim
        self.with_cls_token = True
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.transformer_encoder_decoder = nn.Transformer(
            d_model=model_dim, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers, dim_feedforward=model_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.out_linear_layer = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(model_dim // 2, model_dim)
        )

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs):
        B = inputs.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, inputs), dim=1)
        x = self.transformer_encoder_decoder(inputs, x)
        if self.with_cls_token:
            x = x[:, 0]

        residual = self.out_linear_layer(x)
        x = x + residual

        return x
