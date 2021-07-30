import pytest
import torch

from mmdet.models.utils import ConvUpsample

@pytest.mark.parametrize(
    'num_upsample', [2, 3, 4])
def test_conv_upsample(num_upsample):
    layer = ConvUpsample(
        10,
        5,
        num_layers=3,
        num_upsample=num_upsample,
        conv_cfg=None,
        norm_cfg=None,
    )
