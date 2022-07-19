import os
from pathlib import Path

import numpy as np
import pytest
import torch

from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

# TODO: Waiting to fix multiple call error bug
register_all_modules()


@pytest.mark.parametrize('config,devices',
                         [('configs/retinanet/retinanet_r18_fpn_1x_coco.py',
                           ('cpu', 'cuda'))])
def test_init_detector(config, devices):
    assert all([device in ['cpu', 'cuda'] for device in devices])

    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, '..')

    config_file = os.path.join(project_dir, config)

    # test init_detector with config_file: str and cfg_options
    cfg_options = dict(
        model=dict(
            backbone=dict(
                depth=18,
                init_cfg=dict(
                    type='Pretrained', checkpoint='torchvision://resnet18'))))

    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('test requires GPU and torch+cuda')

        model = init_detector(
            config_file, device=device, cfg_options=cfg_options)

        # test init_detector with :obj:`Path`
        config_path_object = Path(config_file)
        model = init_detector(config_path_object, device=device)

        # test init_detector with undesirable type
        with pytest.raises(TypeError):
            config_list = [config_file]
            model = init_detector(config_list)  # noqa: F841


@pytest.mark.parametrize('config,devices',
                         [('configs/retinanet/retinanet_r18_fpn_1x_coco.py',
                           ('cpu', 'cuda'))])
def test_inference_detector(config, devices):
    assert all([device in ['cpu', 'cuda'] for device in devices])

    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, '..')

    config_file = os.path.join(project_dir, config)

    # test init_detector with config_file: str and cfg_options
    rng = np.random.RandomState(0)
    img1 = rng.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img2 = rng.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    for device in devices:
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip('test requires GPU and torch+cuda')

        model = init_detector(config_file, device=device)
        result = inference_detector(model, img1)
        assert isinstance(result, DetDataSample)
        result = inference_detector(model, [img1, img2])
        assert isinstance(result, list) and len(result) == 2
