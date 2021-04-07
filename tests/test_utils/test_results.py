import copy

import numpy as np
import pytest
import torch

from mmdet.core.results.results import InstanceResults, Results


def test_results():
    meta_info = dict(
        img_size=(256, 256), path='dadfaff', scale_factor=np.array([1.5, 1.5]))

    results = Results(meta_info)

    with pytest.raises(AssertionError):
        delattr(results, '_meta_info_field')

    with pytest.raises(AssertionError):
        delattr(results, '_results_field')

    assert results.device is None
    results.a = 1000
    results['a'] = 2000
    assert results['a'] == 2000
    assert results.a == 2000
    assert results.get('a') == results['a'] == results.a
    results._meta = 1000
    assert '_meta' not in results.keys()
    if torch.cuda.is_available():
        results.bbox = torch.ones(2, 3, 4, 5).cuda()
        results.score = torch.ones(2, 3, 4, 4)
        assert results.score.device == \
               results.bbox.device == results.device
        assert torch.ones(1).to(results.device).device ==\
               results.device
    else:
        results.bbox = torch.ones(2, 3, 4, 5)

    assert results.device == results.bbox.device
    assert len(results.new_results().results()) == 0
    results.bbox.sum()
    with pytest.raises(AttributeError):
        results.img_size = 100

    for k, v in results.items():
        if k == 'bbox':
            assert isinstance(v, torch.Tensor)
    assert results.has('a')
    results.remove('a')
    assert not results.has('a')

    new_results = copy.deepcopy(results)
    new_results.bbox[0] = 100
    assert new_results.bbox.sum() != results.bbox.sum()
    cpu_results = new_results.cpu()
    for k, v in cpu_results.items():
        if isinstance(v, torch.Tensor):
            assert not v.is_cuda

    assert isinstance(cpu_results.numpy().bbox, np.ndarray)
    dump_results = results.export_results()
    assert 'img_size' not in dump_results

    for v in dump_results.values():
        assert not isinstance(v, torch.Tensor)

    copy_results = copy.copy(results)
    assert copy_results.bbox is results.bbox
    deep_copy = copy.deepcopy(results)
    assert deep_copy.bbox is not results.bbox
    double_results = results.to(torch.double)
    for k, v in double_results.items():
        if isinstance(v, torch.Tensor):
            assert v.dtype is torch.double


def test_instanceresults():
    meta_info = dict(
        img_size=(256, 256), path='dadfaff', scale_factor=np.array([1.5, 1.5]))
    results = InstanceResults(meta_info)
    num_instance = 1000

    results_list = []
    for _ in range(2):
        results['bbox'] = torch.rand(num_instance, 4)
        results['label'] = torch.rand(num_instance, 1)
        results['mask'] = torch.rand(num_instance, 224, 224)
        results['instances_infos'] = [1] * num_instance
        results['cpu_bbox'] = np.random.random((num_instance, 4))
        assert len(results[0]) == 1
        with pytest.raises(IndexError):
            return results[num_instance + 1]
        with pytest.raises(AssertionError):
            results.centerness = torch.rand(num_instance + 1, 1)

        mask_tensor = torch.rand(num_instance) > 0.5
        length = mask_tensor.sum()
        assert len(results[mask_tensor]) == length

        index_tensor = torch.LongTensor([1, 5, 8, 110, 399])
        length = len(index_tensor)

        assert len(results[index_tensor]) == length

        results_list.append(results)

    cat_resutls = InstanceResults.cat(results_list)
    assert len(cat_resutls) == num_instance * 2
