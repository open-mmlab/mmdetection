import math

import torch
import torch.nn as nn
from mmcv.cnn import uniform_init


class PositionEmbeddingSine(nn.Module):
    """Position encoding with sine and cosine functions.

    See 'End-to-End Object Detection with Transformers'
    (https://arxiv.org/pdf/2005.12872) for details.

    Args:
        num_pos_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int): The temperature used for calculating the
            position embedding. Default 10000.
        normalize (bool): Whether to do normalization when calculating
            the position embedding. Default False.
        scale (float): A scale factor for calculating the position
            embedding. Default None. If None, 2*pi will be used.
        eps (float): A value added to the denominator for numerical
            stability. Default 1e-6.
    """

    def __init__(self,
                 num_pos_feats,
                 temperature=10000,
                 normalize=False,
                 scale=None,
                 eps=1e-6):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.eps = eps

    def forward(self, mask):
        """Forward function for `PositionEmbeddingSine`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image.

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs,num_pos_feats*2,h,w].
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_pos_feats={self.num_pos_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


class PositionEmbeddingLearned(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_pos_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        num_embed (int): The dictionary size of embeddings.
    """

    def __init__(self, num_pos_feats, num_embed=50):
        super(PositionEmbeddingLearned, self).__init__()
        self.row_embed = nn.Embedding(num_embed, num_pos_feats)
        self.col_embed = nn.Embedding(num_embed, num_pos_feats)
        self.num_pos_feats = num_pos_feats
        self.num_embed = num_embed
        self.init_weights()

    def init_weights(self):
        uniform_init(self.row_embed)
        uniform_init(self.col_embed)

    def forward(self, mask):
        """Forward function for `PositionEmbeddingLearned`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image.

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs,num_pos_feats*2,h,w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_pos_feats={self.num_pos_feats}, '
        repr_str += f'num_embed={self.num_embed})'
        return repr_str
