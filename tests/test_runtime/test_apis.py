import os
import shutil
import urllib
from pathlib import Path

import pytest

from mmdet.apis import init_detector


def test_init_detector():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, '..')

    config_file = os.path.join(
        project_dir, 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py')

    checkpoint_file = os.path.join(
        project_dir,
        'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')

    if not os.path.exists(checkpoint_file):
        url = ('https://download.openmmlab.com/mmdetection/v2.0'
               '/mask_rcnn/mask_rcnn_r50_fpn_1x_coco'
               '/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')
        print(f'Downloading {url} ...')
        local_filename, _ = urllib.request.urlretrieve(url)
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        shutil.move(local_filename, checkpoint_file)
        print(f'Saved as {checkpoint_file}')
    else:
        print(f'Using existing checkpoint {checkpoint_file}')

    # test init_detector with config_file: str and checkpoint
    model = init_detector(
        config_file, checkpoint=checkpoint_file, device='cpu')

    # test init_detector with config_file: str and cfg_options
    cfg_options = dict(
        model=dict(
            backbone=dict(
                depth=101,
                init_cfg=dict(
                    type='Pretrained', checkpoint='torchvision://resnet101'))))
    model = init_detector(config_file, device='cpu', cfg_options=cfg_options)

    # test init_detector with :obj:`Path`
    config_path_object = Path(config_file)
    model = init_detector(config_path_object)

    # test init_detector with undesirable type
    with pytest.raises(TypeError):
        config_int = 45678
        model = init_detector(config_int, device='cpu')  # noqa: F841
