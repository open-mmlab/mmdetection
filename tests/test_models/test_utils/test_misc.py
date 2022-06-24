import pytest
import torch
from mmengine.data import InstanceData

from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.testing import demo_mm_inputs


def test_parse_gt_instance_info():
    packed_inputs = demo_mm_inputs()

    batch_data_samples = []
    for inputs in packed_inputs:
        batch_data_samples.append(inputs['data_sample'])
    batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
        = unpack_gt_instances(batch_data_samples)
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
