# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine.data import BaseDataElement as PixelData
from mmengine.data import InstanceData

from mmdet.core import DetDataSample


def _get_config_directory():
    """Find the predefined detector config directory."""
    try:
        # Assume we are running in the source mmdetection repo
        repo_dpath = dirname(dirname(dirname(__file__)))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmdet
        repo_dpath = dirname(dirname(mmdet.__file__))
    config_dpath = join(repo_dpath, 'configs')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """Load a configuration as a python module."""
    from mmengine import Config
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = Config.fromfile(config_fpath)
    return config_mod


def get_detector_cfg(fname):
    """Grab configs necessary to create a detector.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    return model


def _rand_bboxes(rng, num_boxes, w, h):
    cx, cy, bw, bh = rng.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


def _rand_masks(rng, num_boxes, bboxes, img_w, img_h):
    masks = np.zeros((num_boxes, img_h, img_w))
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return torch.from_numpy(masks)


def demo_mm_inputs(batch_size=2,
                   image_shapes=(3, 128, 128),
                   num_items=None,
                   num_classes=10,
                   with_mask=False,
                   with_semantic=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        batch_size (int): batch size. Defaults to 2.
        image_shapes (List[tuple], Optional): image shape.
            Defaults to (3, 128, 128)
        num_items (None | List[int]): specifies the number
            of boxes in each batch item. Default to None.
        num_classes (int): number of different labels a
            box might have. Defaults to 10.
        with_mask (bool): Whether to return mask annotation.
            Defaults to False.
        with_semantic (bool): whether to return semantic.
            Defaults to False.
    """
    # from mmdet.core import BitmapMasks
    rng = np.random.RandomState(0)

    if isinstance(image_shapes, list):
        assert len(image_shapes) == batch_size
    else:
        image_shapes = [image_shapes] * batch_size

    if isinstance(num_items, list):
        assert len(num_items) == batch_size

    packed_inputs = []
    for idx in range(batch_size):
        image_shape = image_shapes[idx]
        c, h, w = image_shape

        image = rng.randint(0, 255, size=image_shape, dtype=np.uint8)

        mm_inputs = dict()
        mm_inputs['inputs'] = torch.from_numpy(image)

        img_meta = {
            'img_id': idx,
            'img_shape': image_shape[1:],
            'ori_shape': image_shape[1:],
            'filename': '<demo>.png',
            'scale_factor': np.array([1.1, 1.2]),
            'flip': False,
            'flip_direction': None,
            'border': [1, 1, 1, 1]  # Only used by CenterNet
        }

        data_sample = DetDataSample()
        data_sample.set_metainfo(img_meta)

        # gt_instances
        gt_instances = InstanceData()
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[idx]

        bboxes = _rand_bboxes(rng, num_boxes, w, h)
        labels = rng.randint(1, num_classes, size=num_boxes)
        gt_instances.bboxes = torch.FloatTensor(bboxes)
        gt_instances.labels = torch.LongTensor(labels)

        if with_mask:
            masks = _rand_masks(rng, num_boxes, bboxes, w, h)
            gt_instances.masks = masks

        # TODO: waiting for ci to be fixed
        # masks = np.random.randint(0, 2, (len(bboxes), h, w), dtype=np.uint8)
        # gt_instances.mask = BitmapMasks(masks, h, w)

        data_sample.gt_instances = gt_instances

        # ignore_instances
        ignore_instances = InstanceData()
        bboxes = _rand_bboxes(rng, num_boxes, w, h)
        ignore_instances.bboxes = torch.FloatTensor(bboxes)
        data_sample.ignored_instances = ignore_instances

        # gt_sem_seg
        if with_semantic:
            # assume gt_semantic_seg using scale 1/8 of the img
            gt_semantic_seg = torch.from_numpy(
                np.random.randint(
                    0, num_classes, (1, h // 8, w // 8), dtype=np.uint8))
            gt_sem_seg_data = dict(sem_seg=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        mm_inputs['data_sample'] = data_sample

        # TODO: gt_ignore

        packed_inputs.append(mm_inputs)
    return packed_inputs
