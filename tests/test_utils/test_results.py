import numpy as np
import pytest
import torch

from mmdet.core.results.results import InstanceResults, Results


def test_results():
    # test init
    meta_info = dict(
        img_size=[256, 256], path='dadfaff', scale_factor=np.array([1.5, 1.5]))

    results = Results(meta_info)

    with pytest.raises(AssertionError):
        delattr(results, '_meta_info_field')

    with pytest.raises(AssertionError):
        delattr(results, '_results_field')

    for k, v in results.meta_info_field.items():
        if isinstance(v, np.ndarray):
            assert (results.meta_info_field[k] == meta_info[k]).all()
        else:
            assert results.meta_info_field[k] == meta_info[k]

    # test `add_meta_info`
    # attribute in `_meta_info_field` is unmodifiable once initialized
    with pytest.raises(KeyError):
        results.add_meta_info(meta_info)

    new_meta = dict(padding_shape=(1000, 1000))
    results.add_meta_info(new_meta)
    meta_info.update(new_meta)
    for k, v in results.meta_info_field.items():
        if isinstance(v, np.ndarray):
            assert (results.meta_info_field[k] == meta_info[k]).all()
        else:
            assert results.meta_info_field[k] == meta_info[k]
    set(results.meta_info_field.keys()) == set(meta_info.keys())

    # test `new_results`
    new_results = results.new_results()
    assert len(new_results.results_field) == 0
    for k, v in new_results.meta_info_field.items():
        if isinstance(v, np.ndarray):
            assert (results.meta_info_field[k] == new_results[k]).all()
        else:
            assert results.meta_info_field[k] == new_results[k]
    # test deecopy when use image_meta
    new_results.img_size[0] = 0
    assert not new_results.img_size == results.img_size

    # test __setattr__
    new_results.mask = torch.rand(1, 3, 4, 5)
    new_results.bboxes = torch.rand(1, 3, 4, 5)

    # test results_field has been updated
    assert 'mask' in new_results.results_field
    assert 'mask' in new_results._results_field

    # '_meta_info_field', '_results_field' is unmodifiable.
    with pytest.raises(AttributeError):
        new_results._results_field = dict()
    with pytest.raises(AttributeError):
        new_results._results_field = dict()

    # all tensor would be moved to same device
    if torch.cuda.is_available():
        new_results.cuda_det = torch.ones(1, 2, 4, 4, device='cuda')
        assert new_results.device == new_results.cuda_det.device
        assert not new_results.cuda_det.is_cuda
        del new_results.cuda_det

    # test

    # test `in`
    assert 'mask' in new_results

    # test_keys

    # test values

    # test items

    results['_c'] = 10000
    results.get('dad', None) is None
    assert hasattr(results, '_c')
    del results['_c']
    assert not hasattr(results, '_c')

    assert results.device is None
    results.a = 1000
    results['a'] = 2000
    assert results['a'] == 2000
    assert results.a == 2000
    assert results.get('a') == results['a'] == results.a
    results._meta = 1000
    assert '_meta' in results.keys()
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
    assert len(results.new_results().results_field) == 0
    results.bbox.sum()
    with pytest.raises(AttributeError):
        results.img_size = 100

    for k, v in results.items():
        if k == 'bbox':
            assert isinstance(v, torch.Tensor)
    assert 'a' in results
    results.pop('a')
    assert 'a' not in results

    cpu_results = results.cpu()
    for k, v in cpu_results.items():
        if isinstance(v, torch.Tensor):
            assert not v.is_cuda

    assert isinstance(cpu_results.numpy().bbox, np.ndarray)

    cuda_resutls = results.cuda()
    for k, v in cuda_resutls.items():
        if isinstance(v, torch.Tensor):
            assert v.is_cuda

    double_results = results.to(torch.double)
    for k, v in double_results.items():
        if isinstance(v, torch.Tensor):
            assert v.dtype is torch.double


def test_instance_results():
    meta_info = dict(
        img_size=(256, 256),
        path='dadfaff',
        scale_factor=np.array([1.5, 1.5, 1, 1]))
    results = InstanceResults(meta_info)
    num_instance = 1000

    results_list = []
    for _ in range(2):
        results['bbox'] = torch.rand(num_instance, 4)
        results['label'] = torch.rand(num_instance, 1)
        results['mask'] = torch.rand(num_instance, 224, 224)
        results['instances_infos'] = [1] * num_instance
        results['cpu_bbox'] = np.random.random((num_instance, 4))
        if torch.cuda.is_available():
            results.cuda_tensor = torch.rand(num_instance).cuda()
            assert not results.cuda_tensor.is_cuda
            cuda_results = results.cuda()
            assert cuda_results.cuda_tensor.is_cuda

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
