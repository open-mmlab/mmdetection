import copy

import numpy as np
import pytest
import torch

from mmdet.core import GeneralData, InstanceData


def test_results():

    def _equal(a, b):
        if isinstance(a, (torch.Tensor, np.ndarray)):
            return (a == b).all()
        else:
            return a == b

    # test init
    meta_info = dict(
        img_size=[256, 256],
        path='dadfaff',
        scale_factor=np.array([1.5, 1.5]),
        img_shape=torch.rand(4))

    data = dict(
        bboxes=torch.rand(4, 4),
        labels=torch.rand(4),
        masks=np.random.rand(4, 2, 2))

    results = GeneralData(meta_info=meta_info)
    assert 'img_size' in results
    assert results.img_size == [256, 256]
    assert results['img_size'] == [256, 256]
    assert 'path' in results
    assert results.path == 'dadfaff'

    results = GeneralData(meta_info=meta_info, data=dict(bboxes=torch.rand(5)))
    assert 'bboxes' in results
    assert len(results.bboxes) == 5

    # data should be a dict
    with pytest.raises(AssertionError):
        GeneralData(data=1)

    # test set data
    results = GeneralData()
    results.set_data(data)
    assert 'bboxes' in results
    assert len(results.bboxes) == 4
    assert 'masks' in results
    assert len(results.masks) == 4
    # data should be a dict
    with pytest.raises(AssertionError):
        results.set_data(data=1)

    # test set_meta
    results = GeneralData()
    results.set_meta_info(meta_info)
    assert 'img_size' in results
    assert results.img_size == [256, 256]
    assert results['img_size'] == [256, 256]
    assert 'path' in results
    assert results.path == 'dadfaff'
    # can skip same value when overwrite
    results.set_meta_info(meta_info)

    # meta should be a dict
    with pytest.raises(AssertionError):
        results.set_meta_info(meta_info='fjhka')

    # attribute in `_meta_info_field` is immutable once initialized
    results.set_meta_info(meta_info)
    # meta should be immutable
    with pytest.raises(AssertionError):
        GeneralData.set_meta_info(dict(img_size=[254, 251]))
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['path'] = 'dada'
        results.set_meta_info(duplicate_meta_info)
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['scale_factor'] = np.array([1.5, 1.6])
        results.set_meta_info(duplicate_meta_info)

    # test new_results
    results = GeneralData(meta_info)
    new_results = results.new_results()
    for k, v in results.meta_items():
        assert k in new_results
        _equal(v, new_results[k])

    results = GeneralData(meta_info, data=data)
    temp_meta = copy.deepcopy(meta_info)
    temp_data = copy.deepcopy(data)
    temp_data['time'] = '12212'
    temp_meta['img_norm'] = np.random.random(3)

    new_results = results.new_results(meta_info=temp_meta, data=temp_data)
    for k, v in new_results.meta_items():
        if k in results:
            _equal(v, results[k])
        else:
            assert _equal(v, temp_meta[k])
            assert k == 'img_norm'

    for k, v in new_results.items():
        if k in results:
            _equal(v, results[k])
        else:
            assert k == 'time'
            assert _equal(v, temp_data[k])

    # test keys
    results = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'bboxes' in results.keys()
    results.b = 10
    assert 'b' in results

    # test meta keys
    results = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'path' in results.meta_keys()
    assert len(results.meta_keys()) == len(meta_info)
    results.set_meta_info(dict(workdir='fafaf'))
    assert 'workdir' in results
    assert len(results.meta_keys()) == len(meta_info) + 1

    # test values
    results = GeneralData(meta_info, data=dict(bboxes=10))
    assert 10 in results.values()
    assert len(results.values()) == 1

    # test meta values
    results = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'dadfaff' in results.meta_values()
    assert len(results.meta_values()) == len(meta_info)

    # test items
    results = GeneralData(data=data)
    for k, v in results.items():
        assert k in data
        assert _equal(v, data[k])

    # test meta_items
    results = GeneralData(meta_info=meta_info)
    for k, v in results.meta_items():
        assert k in meta_info
        assert _equal(v, meta_info[k])

    # test __setattr__
    new_results = GeneralData(data=data)
    new_results.mask = torch.rand(3, 4, 5)
    new_results.bboxes = torch.rand(2, 4)
    assert 'mask' in new_results
    assert len(new_results.mask) == 3
    assert len(new_results.bboxes) == 2

    # test results_field has been updated
    assert 'mask' in new_results._data_fields
    assert 'bboxes' in new_results._data_fields

    for k in data:
        assert k in new_results._data_fields

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        new_results._data_fields = None
    with pytest.raises(AttributeError):
        new_results._meta_info_fields = None
    with pytest.raises(AttributeError):
        del new_results._data_fields
    with pytest.raises(AttributeError):
        del new_results._meta_info_fields

    # key in _meta_info_field is immutable
    new_results.set_meta_info(meta_info)
    with pytest.raises(KeyError):
        del new_results.img_size
    with pytest.raises(KeyError):
        del new_results.scale_factor
    for k in new_results.meta_keys():
        with pytest.raises(AttributeError):
            new_results[k] = None

    # test __delattr__
    # test key can be removed in results_field
    assert 'mask' in new_results._data_fields
    assert 'mask' in new_results.keys()
    assert 'mask' in new_results
    assert hasattr(new_results, 'mask')
    del new_results.mask
    assert 'mask' not in new_results.keys()
    assert 'mask' not in new_results
    assert 'mask' not in new_results._data_fields
    assert not hasattr(new_results, 'mask')

    # tset __delitem__
    new_results.mask = torch.rand(1, 2, 3)
    assert 'mask' in new_results._data_fields
    assert 'mask' in new_results
    assert hasattr(new_results, 'mask')
    del new_results['mask']
    assert 'mask' not in new_results
    assert 'mask' not in new_results._data_fields
    assert 'mask' not in new_results
    assert not hasattr(new_results, 'mask')

    # test __setitem__
    new_results['mask'] = torch.rand(1, 2, 3)
    assert 'mask' in new_results._data_fields
    assert 'mask' in new_results.keys()
    assert hasattr(new_results, 'mask')

    # test data_fields has been updated
    assert 'mask' in new_results.keys()
    assert 'mask' in new_results._data_fields

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        del new_results['_data_fields']
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

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(KeyError):
        new_results.pop('_data_fields')
    with pytest.raises(KeyError):
        new_results.pop('_meta_info_field')
    # attribute in `_meta_info_field` is immutable
    with pytest.raises(KeyError):
        new_results.pop('img_size')
    # test pop attribute in results_filed
    new_results['mask'] = torch.rand(1, 2, 3)
    new_results.pop('mask')
    # test data_field has been updated
    assert 'mask' not in new_results
    assert 'mask' not in new_results._data_fields
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

    # test device
    new_results = GeneralData()
    if torch.cuda.is_available():
        newnew_results = new_results.new_results()
        devices = ('cpu', 'cuda')
        for i in range(10):
            device = devices[i % 2]
            newnew_results[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_results = newnew_results.cpu()
        for value in newnew_results.values():
            assert not value.is_cuda
        newnew_results = new_results.new_results()
        devices = ('cuda', 'cpu')
        for i in range(10):
            device = devices[i % 2]
            newnew_results[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_results = newnew_results.cuda()
        for value in newnew_results.values():
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
        for value in cuda_results.values():
            assert value.is_cuda
        cpu_results = cuda_results.cpu()
        for value in cpu_results.values():
            assert not value.is_cuda
        cuda_results = cpu_results.cuda()
        for value in cuda_results.values():
            assert value.is_cuda

    # test detach
    grad_results = double_results.new_results()
    grad_results.mask = torch.rand(2, requires_grad=True)
    grad_results.mask_1 = torch.rand(2, requires_grad=True)
    detach_results = grad_results.detach()
    for value in detach_results.values():
        assert not value.requires_grad

    # test numpy
    tensor_results = double_results.new_results()
    tensor_results.mask = torch.rand(2, requires_grad=True)
    tensor_results.mask_1 = torch.rand(2, requires_grad=True)
    numpy_results = tensor_results.numpy()
    for value in numpy_results.values():
        assert isinstance(value, np.ndarray)
    if torch.cuda.is_available():
        tensor_results = double_results.new_results()
        tensor_results.mask = torch.rand(2)
        tensor_results.mask_1 = torch.rand(2)
        tensor_results = tensor_results.cuda()
        numpy_results = tensor_results.numpy()
        for value in numpy_results.values():
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

    assert len(results.new_results().keys()) == 0
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
    results = InstanceData(meta_info)
    assert 'path' in results

    # test __setattr__
    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        results._data_fields = dict()
    with pytest.raises(AttributeError):
        results._data_fields = dict()

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

    # assert the index should in 0 ~ len(results) -1
    with pytest.raises(IndexError):
        new_results[101]

    # assert the index should not be an empty tensor
    new_new_results = new_results.new_results()
    with pytest.raises(AssertionError):
        new_new_results[0]

    # test str
    with pytest.raises(AssertionError):
        results.img_size_dummmy = meta_info['img_size']

    # test slice
    ten_ressults = new_results[:10]
    len(ten_ressults) == 10
    for v in ten_ressults.values():
        assert len(v) == 10

    # test Longtensor
    long_tensor = torch.randint(100, (50, ))
    long_index_results = new_results[long_tensor]
    assert len(long_index_results) == len(long_tensor)
    for key, value in long_index_results.items():
        if not isinstance(value, list):
            assert (long_index_results[key] == new_results[key][long_tensor]
                    ).all()
        else:
            len(long_tensor) == len(value)

    # test bool tensor
    bool_tensor = torch.rand(100) > 0.5
    bool_index_results = new_results[bool_tensor]
    assert len(bool_index_results) == bool_tensor.sum()
    for key, value in bool_index_results.items():
        if not isinstance(value, list):
            assert (bool_index_results[key] == new_results[key][bool_tensor]
                    ).all()
        else:
            assert len(value) == bool_tensor.sum()

    num_instance = 1000
    results_list = []

    # assert len(instance_lists) > 0
    with pytest.raises(AssertionError):
        results.cat(results_list)

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

    cat_resutls = InstanceData.cat(results_list)
    assert len(cat_resutls) == num_instance * 2
