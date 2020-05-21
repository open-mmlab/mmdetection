import os.path as osp

import numpy as np

from mmdet.datasets.pipelines import Albu, LoadImageFromFile, Normalize


def test_albu_transform():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), '../data'),
        img_info=dict(filename='color.jpg'))

    # Define simple pipeline
    load = LoadImageFromFile()
    albu_transform = Albu(transforms=[dict(type='ChannelShuffle', p=1)])
    normalize = Normalize(mean=[0] * 3, std=[0] * 3, to_rgb=True)

    # Execute transforms
    results = load(results)
    results = albu_transform(results)
    results = normalize(results)

    assert results['img'].dtype == np.float32
