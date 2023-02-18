import numpy as np
import torch

from mmdet.core import BitmapMasks
from mmdet.models.utils import preprocess_panoptic_gt


def test_preprocess_panoptic_gt():
    img_metas = {
        'batch_input_shape': (32, 32),
        'pad_shape': (32, 28, 3),
        'img_shape': (30, 27, 3),
    }
    num_things = 8
    num_stuff = 5
    gt_labels = torch.tensor([0, 1], dtype=torch.long)
    gt_masks = np.zeros((2, ) + img_metas['pad_shape'][:2], dtype=np.uint8)
    gt_masks[0, :5, :5] = 1
    gt_masks[1, 5:7] = 1
    gt_masks = BitmapMasks(
        gt_masks,
        width=img_metas['pad_shape'][0],
        height=img_metas['pad_shape'][1])

    # gt_semantic_seg is None and merge_things_stuff is True
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=None,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=True)
    assert len(out) == 4
    assert torch.all(out[0] == gt_labels)
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert out[2] is None and out[3] is None

    # instance only
    # gt_semantic_seg is None and merge_things_stuff is False
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=None,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=False)
    assert len(out) == 4
    assert torch.all(out[0] == gt_labels)
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert out[2] is None and out[3] is None

    # empty stuff
    gt_semantic_seg = torch.full(
        (1, ) + img_metas['batch_input_shape'], 255, dtype=int)
    gt_semantic_seg[0, :5, :5] = 0
    gt_semantic_seg[0, 5:] = 1

    # gt_semantic_seg is Not None but empty, and merge_things_stuff is True
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=gt_semantic_seg,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=True)
    assert len(out) == 4
    assert torch.all(out[0] == gt_labels)
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert out[2] is None and out[3] is None

    # gt_semantic_seg is Not None but empty, and merge_things_stuff is False
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=gt_semantic_seg,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=False)
    assert len(out) == 4
    assert torch.all(out[0] == gt_labels)
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert out[2].shape[0] == 0 and out[3].shape[0] == 0

    gt_semantic_seg = torch.full((1, ) + img_metas['batch_input_shape'], 255)
    gt_semantic_seg[0, :5, :5] = 0
    gt_semantic_seg[0, 5:] = 1
    gt_semantic_seg[0, :, 5:] = num_things + 2

    # gt_semantic_seg is not empty, and merge_things_stuff is True
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=gt_semantic_seg,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=True)
    assert len(out) == 4
    assert torch.all(
        out[0] == torch.tensor([0, 1, num_things + 2], dtype=torch.long))
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert out[2] is None and out[3] is None

    # gt_semantic_seg is not empty, and merge_things_stuff is False
    out = preprocess_panoptic_gt(
        gt_labels=gt_labels,
        gt_masks=gt_masks,
        gt_semantic_seg=gt_semantic_seg,
        num_things=num_things,
        num_stuff=num_stuff,
        img_metas=img_metas,
        merge_things_stuff=False)
    assert len(out) == 4
    assert torch.all(out[0] == gt_labels)
    assert isinstance(
        out[1],
        torch.Tensor) and out[1].shape[-2:] == img_metas['batch_input_shape']
    assert torch.all(
        out[2] == torch.tensor([num_things + 2], dtype=torch.long))
    assert torch.all(out[3] == (gt_semantic_seg == num_things + 2).long())
