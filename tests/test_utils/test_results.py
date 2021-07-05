import numpy as np
import pytest
import torch

from mmdet.core.results.results import InstanceResults, Results


def test_results():
    # test init
    meta_info = dict(
        img_size=[256, 256], path='dadfaff', scale_factor=np.array([1.5, 1.5]))

    results = Results(meta_info)

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

    # attribute releated to meta info is unmodifiable
    with pytest.raises(AttributeError):
        new_results._results_field = dict()

    with pytest.raises(AttributeError):
        new_results.scale_factor = 1

    # '_meta_info_field', '_results_field' is unmodifiable.
    with pytest.raises(AttributeError):
        del new_results._results_field
    with pytest.raises(AttributeError):
        del new_results._meta_info_field

    # key in _meta_info_field is unmodifiable
    with pytest.raises(KeyError):
        del new_results.img_size
    with pytest.raises(KeyError):
        del new_results.scale_factor

    # test key can be removed in results_field
    assert 'mask' in new_results._results_field
    assert 'mask' in new_results.results_field
    assert hasattr(new_results, 'mask')
    del new_results.mask
    assert 'mask' not in new_results
    assert 'mask' not in new_results._results_field
    assert 'mask' not in new_results.results_field
    assert not hasattr(new_results, 'mask')

    # tset __delitem__
    new_results.mask = torch.rand(1, 2, 3)
    assert 'mask' in new_results._results_field
    assert 'mask' in new_results.results_field
    assert hasattr(new_results, 'mask')
    del new_results['mask']
    assert 'mask' not in new_results
    assert 'mask' not in new_results._results_field
    assert 'mask' not in new_results.results_field
    assert not hasattr(new_results, 'mask')

    # test __setitem__
    new_results['mask'] = torch.rand(1, 2, 3)
    assert 'mask' in new_results._results_field
    assert 'mask' in new_results.results_field
    assert hasattr(new_results, 'mask')

    # test results_field has been updated
    assert 'mask' in new_results.results_field
    assert 'mask' in new_results._results_field

    # '_meta_info_field', '_results_field' is unmodifiable.
    with pytest.raises(AttributeError):
        del new_results['_results_field']
    with pytest.raises(AttributeError):
        del new_results['_meta_info_field']

    #  test __getitem__
    new_results.mask is new_results['mask']

    # test get
    assert new_results.get('mask') is new_results.mask
    assert new_results.get('none_attribute', None) is None
    assert new_results.get('none_attribute', 1) == 1

    # test pop
    mask = new_results.mask
    assert new_results.pop('mask') is mask
    assert new_results.pop('mask', None) is None
    assert new_results.pop('mask', 1) == 1

    # '_meta_info_field', '_results_field' is unmodifiable.
    with pytest.raises(KeyError):
        new_results.pop('_results_field')
    with pytest.raises(KeyError):
        new_results.pop('_meta_info_field')
    # attribute in `_meta_info_field` is unmodifiable
    with pytest.raises(KeyError):
        new_results.pop('img_size')
    # test pop attribute in results_filed
    new_results['mask'] = torch.rand(1, 2, 3)
    new_results.pop('mask')
    # test results_field has been updated
    assert 'mask' not in new_results.results_field
    assert 'mask' not in new_results._results_field
    assert 'mask' not in new_results

    # test_keys
    new_results.mask = torch.ones(1, 2, 3)
    'mask' in new_results.keys()
    has_flag = False
    for key in new_results.keys():
        if key == 'mask':
            has_flag = True
    assert has_flag

    # test values
    assert len(list(new_results.keys())) == len(list(new_results.values()))
    mask = new_results.mask
    has_flag = False
    for value in new_results.values():
        if value is mask:
            has_flag = True
    assert has_flag

    # test items
    assert len(list(new_results.keys())) == len(list(new_results.items()))
    mask = new_results.mask
    has_flag = False
    for key, value in new_results.items():
        if value is mask:
            assert key == 'mask'
            has_flag = True
    assert has_flag

    # test results_filed
    results_field = new_results.results_field
    assert len(results_field) == len(new_results.results_field)
    for key in new_results._results_field:
        assert new_results[key] is results_field[key]

    # test meta_file
    meta_info_field = new_results.meta_info_field
    assert len(results_field) == len(new_results.results_field)
    for key in new_results._meta_info_field:
        if isinstance(new_results[key], np.ndarray):
            assert (new_results[key] == meta_info_field[key]).all()
        else:
            assert (new_results[key] == meta_info_field[key])
    # test deep copy to avoid being modified outside
    meta_info_field['img_size'][0] = 100
    assert not meta_info_field['img_size'] == new_results.img_size

    # test device
    if torch.cuda.is_available():
        newnew_results = new_results.new_results()
        devices = ('cpu', 'cuda')
        for i in range(10):
            device = devices[i % 2]
            newnew_results[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_results = newnew_results.cpu()
        for value in newnew_results.results_field.values():
            assert not value.is_cuda
        newnew_results = new_results.new_results()
        devices = ('cuda', 'cpu')
        for i in range(10):
            device = devices[i % 2]
            newnew_results[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_results = newnew_results.cuda()
        for value in newnew_results.results_field.values():
            assert value.is_cuda
    # test to
    double_results = results.new_results()
    double_results.long = torch.LongTensor(1, 2, 3, 4)
    double_results.bool = torch.BoolTensor(1, 2, 3, 4)
    double_results = results.to(torch.double)
    for k, v in double_results.items():
        if isinstance(v, torch.Tensor):
            assert v.dtype is torch.double

    # test .cpu() .cuda()
    if torch.cuda.is_available():
        cpu_results = double_results.new_results()
        cpu_results.mask = torch.rand(1)
        cuda_tensor = torch.rand(1, 2, 3).cuda()
        cuda_results = cpu_results.to(cuda_tensor.device)
        for value in cuda_results.results_field.values():
            assert value.is_cuda
        cpu_results = cuda_results.cpu()
        for value in cpu_results.results_field.values():
            assert not value.is_cuda
        cuda_results = cpu_results.cuda()
        for value in cuda_results.results_field.values():
            assert value.is_cuda

    # test detach
    grad_results = cpu_results.new_results()
    grad_results.mask = torch.rand(2, requires_grad=True)
    grad_results.mask_1 = torch.rand(2, requires_grad=True)
    detach_results = grad_results.detach()
    for value in detach_results.results_field.values():
        assert not value.requires_grad

    # test numpy
    tensor_results = cpu_results.new_results()
    tensor_results.mask = torch.rand(2, requires_grad=True)
    tensor_results.mask_1 = torch.rand(2, requires_grad=True)
    numpy_results = tensor_results.numpy()
    for value in numpy_results.results_field.values():
        assert isinstance(value, np.ndarray)
    if torch.cuda.is_available():
        tensor_results = cpu_results.new_results()
        tensor_results.mask = torch.rand(2)
        tensor_results.mask_1 = torch.rand(2)
        tensor_results = tensor_results.cuda()
        numpy_results = tensor_results.numpy()
        for value in numpy_results.results_field.values():
            assert isinstance(value, np.ndarray)

    results['_c'] = 10000
    results.get('dad', None) is None
    assert hasattr(results, '_c')
    del results['_c']
    assert not hasattr(results, '_c')
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
    else:
        results.bbox = torch.ones(2, 3, 4, 5)

    assert len(results.new_results().results_field) == 0
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

    if torch.cuda.is_available():
        cuda_resutls = results.cuda()
        for k, v in cuda_resutls.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda


def test_instance_results():
    meta_info = dict(
        img_size=(256, 256),
        path='dadfaff',
        scale_factor=np.array([1.5, 1.5, 1, 1]))

    # test init
    results = InstanceResults(meta_info)
    # test deep copy
    assert results.meta_info_field is not meta_info
    assert 'img_size' in results

    # test __setattr__
    # '_meta_info_field', '_results_field' is unmodifiable.
    with pytest.raises(AttributeError):
        results._results_field = dict()
    with pytest.raises(AttributeError):
        results._results_field = dict()

    # all attribute in results_field should be
    # (torch.Tensor, np.ndarray, list))
    with pytest.raises(AssertionError):
        results.a = 1000

    # results field should has same length
    new_results = results.new_results()
    new_results.det_bbox = torch.rand(100, 4)
    new_results.det_label = torch.arange(100)
    with pytest.raises(AssertionError):
        new_results.scores = torch.rand(101, 1)
    new_results.none = [None] * 100
    with pytest.raises(AssertionError):
        new_results.scores = [None] * 101
    new_results.numpy_det = np.random.random([100, 1])
    with pytest.raises(AssertionError):
        new_results.scores = np.random.random([101, 1])

    # isinstance(str, slice, int, torch.LongTensor, torch.BoolTensor)
    item = torch.Tensor([1, 2, 3, 4])
    with pytest.raises(AssertionError):
        new_results[item]
    len(new_results[item.long()]) == 1

    # when input is a bool tensor, The shape of
    # the input at index 0 should equal to
    # the value length in results_field
    with pytest.raises(AssertionError):
        new_results[item.bool()]

    for i in range(len(new_results)):
        assert new_results[i].det_label == i
        assert len(new_results[i]) == 1

    # asset the index should in 0 ~ len(results) -1

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
            assert results.cuda_tensor.is_cuda
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
