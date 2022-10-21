import copy

import pytest
import torch
from mmengine.structures import InstanceData

from mmdet.models.utils import (empty_instances, filter_gt_instances,
                                rename_loss_dict, reweight_loss_dict,
                                unpack_gt_instances)
from mmdet.testing import demo_mm_inputs


def test_parse_gt_instance_info():
    packed_inputs = demo_mm_inputs()['data_samples']
    batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
        = unpack_gt_instances(packed_inputs)
    assert len(batch_gt_instances) == len(packed_inputs)
    assert len(batch_gt_instances_ignore) == len(packed_inputs)
    assert len(batch_img_metas) == len(packed_inputs)


def test_process_empty_roi():
    batch_size = 2
    batch_img_metas = [{'ori_shape': (10, 12)}] * batch_size
    device = torch.device('cpu')

    results_list = empty_instances(batch_img_metas, device, task_type='bbox')
    assert len(results_list) == batch_size
    for results in results_list:
        assert isinstance(results, InstanceData)
        assert len(results) == 0
        assert torch.allclose(results.bboxes, torch.zeros(0, 4, device=device))

    results_list = empty_instances(
        batch_img_metas,
        device,
        task_type='mask',
        instance_results=results_list,
        mask_thr_binary=0.5)
    assert len(results_list) == batch_size
    for results in results_list:
        assert isinstance(results, InstanceData)
        assert len(results) == 0
        assert results.masks.shape == (0, 10, 12)

    # batch_img_metas and instance_results length must be the same
    with pytest.raises(AssertionError):
        empty_instances(
            batch_img_metas,
            device,
            task_type='mask',
            instance_results=[results_list[0]] * 3)


def test_filter_gt_instances():
    packed_inputs = demo_mm_inputs()['data_samples']
    score_thr = 0.7
    with pytest.raises(AssertionError):
        filter_gt_instances(packed_inputs, score_thr=score_thr)

    # filter no instances by score
    for inputs in packed_inputs:
        inputs.gt_instances.scores = torch.ones_like(
            inputs.gt_instances.labels).float()
    filtered_packed_inputs = filter_gt_instances(
        copy.deepcopy(packed_inputs), score_thr=score_thr)
    for filtered_inputs, inputs in zip(filtered_packed_inputs, packed_inputs):
        assert len(filtered_inputs.gt_instances) == len(inputs.gt_instances)

    # filter all instances
    for inputs in packed_inputs:
        inputs.gt_instances.scores = torch.zeros_like(
            inputs.gt_instances.labels).float()
    filtered_packed_inputs = filter_gt_instances(
        copy.deepcopy(packed_inputs), score_thr=score_thr)
    for filtered_inputs in filtered_packed_inputs:
        assert len(filtered_inputs.gt_instances) == 0

    packed_inputs = demo_mm_inputs()['data_samples']
    # filter no instances by size
    wh_thr = (0, 0)
    filtered_packed_inputs = filter_gt_instances(
        copy.deepcopy(packed_inputs), wh_thr=wh_thr)
    for filtered_inputs, inputs in zip(filtered_packed_inputs, packed_inputs):
        assert len(filtered_inputs.gt_instances) == len(inputs.gt_instances)

    # filter all instances by size
    for inputs in packed_inputs:
        img_shape = inputs.img_shape
        wh_thr = (max(wh_thr[0], img_shape[0]), max(wh_thr[1], img_shape[1]))
    filtered_packed_inputs = filter_gt_instances(
        copy.deepcopy(packed_inputs), wh_thr=wh_thr)
    for filtered_inputs in filtered_packed_inputs:
        assert len(filtered_inputs.gt_instances) == 0


def test_rename_loss_dict():
    prefix = 'sup_'
    losses = {'cls_loss': torch.tensor(2.), 'reg_loss': torch.tensor(1.)}
    sup_losses = rename_loss_dict(prefix, losses)
    for name in losses.keys():
        assert sup_losses[prefix + name] == losses[name]


def test_reweight_loss_dict():
    weight = 4
    losses = {'cls_loss': torch.tensor(2.), 'reg_loss': torch.tensor(1.)}
    weighted_losses = reweight_loss_dict(copy.deepcopy(losses), weight)
    for name in losses.keys():
        assert weighted_losses[name] == losses[name] * weight
