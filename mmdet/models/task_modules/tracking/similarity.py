# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor


def embed_similarity(key_embeds: Tensor,
                     ref_embeds: Tensor,
                     method: str = 'dot_product',
                     temperature: int = -1) -> Tensor:
    """Calculate feature similarity from embeddings.

    Args:
        key_embeds (Tensor): Shape (N1, C).
        ref_embeds (Tensor): Shape (N2, C).
        method (str, optional): Method to calculate the similarity,
            options are 'dot_product' and 'cosine'. Defaults to
            'dot_product'.
        temperature (int, optional): Softmax temperature. Defaults to -1.

    Returns:
        Tensor: Similarity matrix of shape (N1, N2).
    """
    assert method in ['dot_product', 'cosine']

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

    similarity = torch.mm(key_embeds, ref_embeds.T)

    if temperature > 0:
        similarity /= float(temperature)
    return similarity
