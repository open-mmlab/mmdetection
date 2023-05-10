# Copyright (c) OpenMMLab. All rights reserved.
import copy
from os.path import dirname, exists, join

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.structures import InstanceData, PixelData

from mmdet.utils.util_random import ensure_rng
from ..registry import TASK_UTILS
from ..structures import DetDataSample, TrackDataSample
from ..structures.bbox import HorizontalBoxes


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


def get_roi_head_cfg(fname):
    """Grab configs necessary to create a roi_head.

    These are deep copied to allow for safe modification of parameters without
    influencing other tests.
    """
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)

    roi_head = model.roi_head
    train_cfg = None if model.train_cfg is None else model.train_cfg.rcnn
    test_cfg = None if model.test_cfg is None else model.test_cfg.rcnn
    roi_head.update(dict(train_cfg=train_cfg, test_cfg=test_cfg))
    return roi_head


def _rand_bboxes(rng, num_boxes, w, h):
    cx, cy, bw, bh = rng.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


def _rand_masks(rng, num_boxes, bboxes, img_w, img_h):
    from mmdet.structures.mask import BitmapMasks
    masks = np.zeros((num_boxes, img_h, img_w))
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        mask = (rng.rand(1, bbox[3] - bbox[1], bbox[2] - bbox[0]) >
                0.3).astype(np.int64)
        masks[i:i + 1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask
    return BitmapMasks(masks, height=img_h, width=img_w)


def demo_mm_inputs(batch_size=2,
                   image_shapes=(3, 128, 128),
                   num_items=None,
                   num_classes=10,
                   sem_seg_output_strides=1,
                   with_mask=False,
                   with_semantic=False,
                   use_box_type=False,
                   device='cpu'):
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
        device (str): Destination device type. Defaults to cpu.
    """
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
        mm_inputs['inputs'] = torch.from_numpy(image).to(device)

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
        # TODO: remove this part when all model adapted with BaseBoxes
        if use_box_type:
            gt_instances.bboxes = HorizontalBoxes(bboxes, dtype=torch.float32)
        else:
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
        if use_box_type:
            ignore_instances.bboxes = HorizontalBoxes(
                bboxes, dtype=torch.float32)
        else:
            ignore_instances.bboxes = torch.FloatTensor(bboxes)
        data_sample.ignored_instances = ignore_instances

        # gt_sem_seg
        if with_semantic:
            # assume gt_semantic_seg using scale 1/8 of the img
            gt_semantic_seg = torch.from_numpy(
                np.random.randint(
                    0,
                    num_classes, (1, h // sem_seg_output_strides,
                                  w // sem_seg_output_strides),
                    dtype=np.uint8))
            gt_sem_seg_data = dict(sem_seg=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        mm_inputs['data_samples'] = data_sample.to(device)

        # TODO: gt_ignore

        packed_inputs.append(mm_inputs)
    data = pseudo_collate(packed_inputs)
    return data


def demo_mm_proposals(image_shapes, num_proposals, device='cpu'):
    """Create a list of fake porposals.

    Args:
        image_shapes (list[tuple[int]]): Batch image shapes.
        num_proposals (int): The number of fake proposals.
    """
    rng = np.random.RandomState(0)

    results = []
    for img_shape in image_shapes:
        result = InstanceData()
        w, h = img_shape[1:]
        proposals = _rand_bboxes(rng, num_proposals, w, h)
        result.bboxes = torch.from_numpy(proposals).float()
        result.scores = torch.from_numpy(rng.rand(num_proposals)).float()
        result.labels = torch.zeros(num_proposals).long()
        results.append(result.to(device))
    return results


def demo_mm_sampling_results(proposals_list,
                             batch_gt_instances,
                             batch_gt_instances_ignore=None,
                             assigner_cfg=None,
                             sampler_cfg=None,
                             feats=None):
    """Create sample results that can be passed to BBoxHead.get_targets."""
    assert len(proposals_list) == len(batch_gt_instances)
    if batch_gt_instances_ignore is None:
        batch_gt_instances_ignore = [None for _ in batch_gt_instances]
    else:
        assert len(batch_gt_instances_ignore) == len(batch_gt_instances)

    default_assigner_cfg = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1)
    assigner_cfg = assigner_cfg if assigner_cfg is not None \
        else default_assigner_cfg
    default_sampler_cfg = dict(
        type='RandomSampler',
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=True)
    sampler_cfg = sampler_cfg if sampler_cfg is not None \
        else default_sampler_cfg
    bbox_assigner = TASK_UTILS.build(assigner_cfg)
    bbox_sampler = TASK_UTILS.build(sampler_cfg)

    sampling_results = []
    for i in range(len(batch_gt_instances)):
        if feats is not None:
            feats = [lvl_feat[i][None] for lvl_feat in feats]
        # rename proposals.bboxes to proposals.priors
        proposals = proposals_list[i]
        proposals.priors = proposals.pop('bboxes')

        assign_result = bbox_assigner.assign(proposals, batch_gt_instances[i],
                                             batch_gt_instances_ignore[i])
        sampling_result = bbox_sampler.sample(
            assign_result, proposals, batch_gt_instances[i], feats=feats)
        sampling_results.append(sampling_result)

    return sampling_results


def demo_track_inputs(batch_size=1,
                      num_frames=2,
                      key_frames_inds=None,
                      image_shapes=(3, 128, 128),
                      num_items=None,
                      num_classes=1,
                      with_mask=False,
                      with_semantic=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        batch_size (int): batch size. Default to 1.
        num_frames (int): The number of frames.
        key_frames_inds (List): The indices of key frames.
        image_shapes (List[tuple], Optional): image shape.
            Default to (3, 128, 128)
        num_items (None | List[int]): specifies the number
            of boxes in each batch item. Default to None.
        num_classes (int): number of different labels a
            box might have. Default to 1.
        with_mask (bool): Whether to return mask annotation.
            Defaults to False.
        with_semantic (bool): whether to return semantic.
            Default to False.
    """
    rng = np.random.RandomState(0)

    # Make sure the length of image_shapes is equal to ``batch_size``
    if isinstance(image_shapes, list):
        assert len(image_shapes) == batch_size
    else:
        image_shapes = [image_shapes] * batch_size

    packed_inputs = []
    for idx in range(batch_size):
        mm_inputs = dict(inputs=dict())
        _, h, w = image_shapes[idx]

        imgs = rng.randint(
            0, 255, size=(num_frames, *image_shapes[idx]), dtype=np.uint8)
        mm_inputs['inputs'] = torch.from_numpy(imgs)

        img_meta = {
            'img_id': idx,
            'img_shape': image_shapes[idx][-2:],
            'ori_shape': image_shapes[idx][-2:],
            'filename': '<demo>.png',
            'scale_factor': np.array([1.1, 1.2]),
            'flip': False,
            'flip_direction': None,
            'is_video_data': True,
        }

        video_data_samples = []
        for i in range(num_frames):
            data_sample = DetDataSample()
            img_meta['frame_id'] = i
            data_sample.set_metainfo(img_meta)

            # gt_instances
            gt_instances = InstanceData()
            if num_items is None:
                num_boxes = rng.randint(1, 10)
            else:
                num_boxes = num_items[idx]

            bboxes = _rand_bboxes(rng, num_boxes, w, h)
            labels = rng.randint(0, num_classes, size=num_boxes)
            instances_id = rng.randint(100, num_classes + 100, size=num_boxes)
            gt_instances.bboxes = torch.FloatTensor(bboxes)
            gt_instances.labels = torch.LongTensor(labels)
            gt_instances.instances_ids = torch.LongTensor(instances_id)

            if with_mask:
                masks = _rand_masks(rng, num_boxes, bboxes, w, h)
                gt_instances.masks = masks

            data_sample.gt_instances = gt_instances
            # ignore_instances
            ignore_instances = InstanceData()
            bboxes = _rand_bboxes(rng, num_boxes, w, h)
            ignore_instances.bboxes = bboxes
            data_sample.ignored_instances = ignore_instances

            video_data_samples.append(data_sample)

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = video_data_samples
        if key_frames_inds is not None:
            assert isinstance(
                key_frames_inds,
                list) and len(key_frames_inds) < num_frames and max(
                    key_frames_inds) < num_frames
            ref_frames_inds = [
                i for i in range(num_frames) if i not in key_frames_inds
            ]
            track_data_sample.set_metainfo(
                dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(
                dict(ref_frames_inds=ref_frames_inds))
        mm_inputs['data_samples'] = track_data_sample

        # TODO: gt_ignore
        packed_inputs.append(mm_inputs)
    data = pseudo_collate(packed_inputs)
    return data


def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``
    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.
    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390 # noqa: E501
    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes


# TODO: Support full ceph
def replace_to_ceph(cfg):
    backend_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/': 's3://openmmlab/datasets/detection/',
            'data/': 's3://openmmlab/datasets/detection/'
        }))

    # TODO: name is a reserved interface, which will be used later.
    def _process_pipeline(dataset, name):

        def replace_img(pipeline):
            if pipeline['type'] == 'LoadImageFromFile':
                pipeline['backend_args'] = backend_args

        def replace_ann(pipeline):
            if pipeline['type'] == 'LoadAnnotations' or pipeline[
                    'type'] == 'LoadPanopticAnnotations':
                pipeline['backend_args'] = backend_args

        if 'pipeline' in dataset:
            replace_img(dataset.pipeline[0])
            replace_ann(dataset.pipeline[1])
            if 'dataset' in dataset:
                # dataset wrapper
                replace_img(dataset.dataset.pipeline[0])
                replace_ann(dataset.dataset.pipeline[1])
        else:
            # dataset wrapper
            replace_img(dataset.dataset.pipeline[0])
            replace_ann(dataset.dataset.pipeline[1])

    def _process_evaluator(evaluator, name):
        if evaluator['type'] == 'CocoPanopticMetric':
            evaluator['backend_args'] = backend_args

    # half ceph
    _process_pipeline(cfg.train_dataloader.dataset, cfg.filename)
    _process_pipeline(cfg.val_dataloader.dataset, cfg.filename)
    _process_pipeline(cfg.test_dataloader.dataset, cfg.filename)
    _process_evaluator(cfg.val_evaluator, cfg.filename)
    _process_evaluator(cfg.test_evaluator, cfg.filename)
