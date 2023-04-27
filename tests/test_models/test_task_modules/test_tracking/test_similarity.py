import torch

from mmdet.models.task_modules import embed_similarity


def test_embed_similarity():
    """Test embed similarity."""
    embeds = torch.rand(2, 3)
    similarity = embed_similarity(embeds, embeds)
    assert similarity.shape == (2, 2)
