import copy

import numpy as np
import pytest
import torch

from mmdet.core import GeneralData, InstanceData


def _equal(a, b):
    if isinstance(a, (torch.Tensor, np.ndarray)):
        return (a == b).all()
    else:
        return a == b


def test_general_data():

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

    instance_data = GeneralData(meta_info=meta_info)
    assert 'img_size' in instance_data
    assert instance_data.img_size == [256, 256]
    assert instance_data['img_size'] == [256, 256]
    assert 'path' in instance_data
    assert instance_data.path == 'dadfaff'

    # test nice_repr
    repr_instance_data = instance_data.new(data=data)
    nice_repr = str(repr_instance_data)
    for line in nice_repr.split('\n'):
        if 'masks' in line:
            assert 'shape' in line
            assert '(4, 2, 2)' in line
        if 'bboxes' in line:
            assert 'shape' in line
            assert 'torch.Size([4, 4])' in line
        if 'path' in line:
            assert 'dadfaff' in line
        if 'scale_factor' in line:
            assert '[1.5 1.5]' in line

    instance_data = GeneralData(
        meta_info=meta_info, data=dict(bboxes=torch.rand(5)))
    assert 'bboxes' in instance_data
    assert len(instance_data.bboxes) == 5

    # data should be a dict
    with pytest.raises(AssertionError):
        GeneralData(data=1)

    # test set data
    instance_data = GeneralData()
    instance_data.set_data(data)
    assert 'bboxes' in instance_data
    assert len(instance_data.bboxes) == 4
    assert 'masks' in instance_data
    assert len(instance_data.masks) == 4
    # data should be a dict
    with pytest.raises(AssertionError):
        instance_data.set_data(data=1)

    # test set_meta
    instance_data = GeneralData()
    instance_data.set_meta_info(meta_info)
    assert 'img_size' in instance_data
    assert instance_data.img_size == [256, 256]
    assert instance_data['img_size'] == [256, 256]
    assert 'path' in instance_data
    assert instance_data.path == 'dadfaff'
    # can skip same value when overwrite
    instance_data.set_meta_info(meta_info)

    # meta should be a dict
    with pytest.raises(AssertionError):
        instance_data.set_meta_info(meta_info='fjhka')

    # attribute in `_meta_info_field` is immutable once initialized
    instance_data.set_meta_info(meta_info)
    # meta should be immutable
    with pytest.raises(KeyError):
        instance_data.set_meta_info(dict(img_size=[254, 251]))
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['path'] = 'dada'
        instance_data.set_meta_info(duplicate_meta_info)
    with pytest.raises(KeyError):
        duplicate_meta_info = copy.deepcopy(meta_info)
        duplicate_meta_info['scale_factor'] = np.array([1.5, 1.6])
        instance_data.set_meta_info(duplicate_meta_info)

    # test new_instance_data
    instance_data = GeneralData(meta_info)
    new_instance_data = instance_data.new()
    for k, v in instance_data.meta_info_items():
        assert k in new_instance_data
        _equal(v, new_instance_data[k])

    instance_data = GeneralData(meta_info, data=data)
    temp_meta = copy.deepcopy(meta_info)
    temp_data = copy.deepcopy(data)
    temp_data['time'] = '12212'
    temp_meta['img_norm'] = np.random.random(3)

    new_instance_data = instance_data.new(meta_info=temp_meta, data=temp_data)
    for k, v in new_instance_data.meta_info_items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert _equal(v, temp_meta[k])
            assert k == 'img_norm'

    for k, v in new_instance_data.items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert k == 'time'
            assert _equal(v, temp_data[k])

    # test keys
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'bboxes' in instance_data.keys()
    instance_data.b = 10
    assert 'b' in instance_data

    # test meta keys
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 'path' in instance_data.meta_info_keys()
    assert len(instance_data.meta_info_keys()) == len(meta_info)
    instance_data.set_meta_info(dict(workdir='fafaf'))
    assert 'workdir' in instance_data
    assert len(instance_data.meta_info_keys()) == len(meta_info) + 1

    # test values
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    assert 10 in instance_data.values()
    assert len(instance_data.values()) == 1

    # test meta values
    instance_data = GeneralData(meta_info, data=dict(bboxes=10))
    # torch 1.3 eq() can not compare str and tensor
    from mmdet import digit_version
    if digit_version(torch.__version__) >= [1, 4]:
        assert 'dadfaff' in instance_data.meta_info_values()
    assert len(instance_data.meta_info_values()) == len(meta_info)

    # test items
    instance_data = GeneralData(data=data)
    for k, v in instance_data.items():
        assert k in data
        assert _equal(v, data[k])

    # test meta_info_items
    instance_data = GeneralData(meta_info=meta_info)
    for k, v in instance_data.meta_info_items():
        assert k in meta_info
        assert _equal(v, meta_info[k])

    # test __setattr__
    new_instance_data = GeneralData(data=data)
    new_instance_data.mask = torch.rand(3, 4, 5)
    new_instance_data.bboxes = torch.rand(2, 4)
    assert 'mask' in new_instance_data
    assert len(new_instance_data.mask) == 3
    assert len(new_instance_data.bboxes) == 2

    # test instance_data_field has been updated
    assert 'mask' in new_instance_data._data_fields
    assert 'bboxes' in new_instance_data._data_fields

    for k in data:
        assert k in new_instance_data._data_fields

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        new_instance_data._data_fields = None
    with pytest.raises(AttributeError):
        new_instance_data._meta_info_fields = None
    with pytest.raises(AttributeError):
        del new_instance_data._data_fields
    with pytest.raises(AttributeError):
        del new_instance_data._meta_info_fields

    # key in _meta_info_field is immutable
    new_instance_data.set_meta_info(meta_info)
    with pytest.raises(KeyError):
        del new_instance_data.img_size
    with pytest.raises(KeyError):
        del new_instance_data.scale_factor
    for k in new_instance_data.meta_info_keys():
        with pytest.raises(AttributeError):
            new_instance_data[k] = None

    # test __delattr__
    # test key can be removed in instance_data_field
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data.keys()
    assert 'mask' in new_instance_data
    assert hasattr(new_instance_data, 'mask')
    del new_instance_data.mask
    assert 'mask' not in new_instance_data.keys()
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert not hasattr(new_instance_data, 'mask')

    # tset __delitem__
    new_instance_data.mask = torch.rand(1, 2, 3)
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data
    assert hasattr(new_instance_data, 'mask')
    del new_instance_data['mask']
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert 'mask' not in new_instance_data
    assert not hasattr(new_instance_data, 'mask')

    # test __setitem__
    new_instance_data['mask'] = torch.rand(1, 2, 3)
    assert 'mask' in new_instance_data._data_fields
    assert 'mask' in new_instance_data.keys()
    assert hasattr(new_instance_data, 'mask')

    # test data_fields has been updated
    assert 'mask' in new_instance_data.keys()
    assert 'mask' in new_instance_data._data_fields

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        del new_instance_data['_data_fields']
    with pytest.raises(AttributeError):
        del new_instance_data['_meta_info_field']

    #  test __getitem__
    new_instance_data.mask is new_instance_data['mask']

    # test get
    assert new_instance_data.get('mask') is new_instance_data.mask
    assert new_instance_data.get('none_attribute', None) is None
    assert new_instance_data.get('none_attribute', 1) == 1

    # test pop
    mask = new_instance_data.mask
    assert new_instance_data.pop('mask') is mask
    assert new_instance_data.pop('mask', None) is None
    assert new_instance_data.pop('mask', 1) == 1

    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(KeyError):
        new_instance_data.pop('_data_fields')
    with pytest.raises(KeyError):
        new_instance_data.pop('_meta_info_field')
    # attribute in `_meta_info_field` is immutable
    with pytest.raises(KeyError):
        new_instance_data.pop('img_size')
    # test pop attribute in instance_data_filed
    new_instance_data['mask'] = torch.rand(1, 2, 3)
    new_instance_data.pop('mask')
    # test data_field has been updated
    assert 'mask' not in new_instance_data
    assert 'mask' not in new_instance_data._data_fields
    assert 'mask' not in new_instance_data

    # test_keys
    new_instance_data.mask = torch.ones(1, 2, 3)
    'mask' in new_instance_data.keys()
    has_flag = False
    for key in new_instance_data.keys():
        if key == 'mask':
            has_flag = True
    assert has_flag

    # test values
    assert len(list(new_instance_data.keys())) == len(
        list(new_instance_data.values()))
    mask = new_instance_data.mask
    has_flag = False
    for value in new_instance_data.values():
        if value is mask:
            has_flag = True
    assert has_flag

    # test items
    assert len(list(new_instance_data.keys())) == len(
        list(new_instance_data.items()))
    mask = new_instance_data.mask
    has_flag = False
    for key, value in new_instance_data.items():
        if value is mask:
            assert key == 'mask'
            has_flag = True
    assert has_flag

    # test device
    new_instance_data = GeneralData()
    if torch.cuda.is_available():
        newnew_instance_data = new_instance_data.new()
        devices = ('cpu', 'cuda')
        for i in range(10):
            device = devices[i % 2]
            newnew_instance_data[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_instance_data = newnew_instance_data.cpu()
        for value in newnew_instance_data.values():
            assert not value.is_cuda
        newnew_instance_data = new_instance_data.new()
        devices = ('cuda', 'cpu')
        for i in range(10):
            device = devices[i % 2]
            newnew_instance_data[f'{i}'] = torch.rand(1, 2, 3, device=device)
        newnew_instance_data = newnew_instance_data.cuda()
        for value in newnew_instance_data.values():
            assert value.is_cuda
    # test to
    double_instance_data = instance_data.new()
    double_instance_data.long = torch.LongTensor(1, 2, 3, 4)
    double_instance_data.bool = torch.BoolTensor(1, 2, 3, 4)
    double_instance_data = instance_data.to(torch.double)
    for k, v in double_instance_data.items():
        if isinstance(v, torch.Tensor):
            assert v.dtype is torch.double

    # test .cpu() .cuda()
    if torch.cuda.is_available():
        cpu_instance_data = double_instance_data.new()
        cpu_instance_data.mask = torch.rand(1)
        cuda_tensor = torch.rand(1, 2, 3).cuda()
        cuda_instance_data = cpu_instance_data.to(cuda_tensor.device)
        for value in cuda_instance_data.values():
            assert value.is_cuda
        cpu_instance_data = cuda_instance_data.cpu()
        for value in cpu_instance_data.values():
            assert not value.is_cuda
        cuda_instance_data = cpu_instance_data.cuda()
        for value in cuda_instance_data.values():
            assert value.is_cuda

    # test detach
    grad_instance_data = double_instance_data.new()
    grad_instance_data.mask = torch.rand(2, requires_grad=True)
    grad_instance_data.mask_1 = torch.rand(2, requires_grad=True)
    detach_instance_data = grad_instance_data.detach()
    for value in detach_instance_data.values():
        assert not value.requires_grad

    # test numpy
    tensor_instance_data = double_instance_data.new()
    tensor_instance_data.mask = torch.rand(2, requires_grad=True)
    tensor_instance_data.mask_1 = torch.rand(2, requires_grad=True)
    numpy_instance_data = tensor_instance_data.numpy()
    for value in numpy_instance_data.values():
        assert isinstance(value, np.ndarray)
    if torch.cuda.is_available():
        tensor_instance_data = double_instance_data.new()
        tensor_instance_data.mask = torch.rand(2)
        tensor_instance_data.mask_1 = torch.rand(2)
        tensor_instance_data = tensor_instance_data.cuda()
        numpy_instance_data = tensor_instance_data.numpy()
        for value in numpy_instance_data.values():
            assert isinstance(value, np.ndarray)

    instance_data['_c'] = 10000
    instance_data.get('dad', None) is None
    assert hasattr(instance_data, '_c')
    del instance_data['_c']
    assert not hasattr(instance_data, '_c')
    instance_data.a = 1000
    instance_data['a'] = 2000
    assert instance_data['a'] == 2000
    assert instance_data.a == 2000
    assert instance_data.get('a') == instance_data['a'] == instance_data.a
    instance_data._meta = 1000
    assert '_meta' in instance_data.keys()
    if torch.cuda.is_available():
        instance_data.bbox = torch.ones(2, 3, 4, 5).cuda()
        instance_data.score = torch.ones(2, 3, 4, 4)
    else:
        instance_data.bbox = torch.ones(2, 3, 4, 5)

    assert len(instance_data.new().keys()) == 0
    with pytest.raises(AttributeError):
        instance_data.img_size = 100

    for k, v in instance_data.items():
        if k == 'bbox':
            assert isinstance(v, torch.Tensor)
    assert 'a' in instance_data
    instance_data.pop('a')
    assert 'a' not in instance_data

    cpu_instance_data = instance_data.cpu()
    for k, v in cpu_instance_data.items():
        if isinstance(v, torch.Tensor):
            assert not v.is_cuda

    assert isinstance(cpu_instance_data.numpy().bbox, np.ndarray)

    if torch.cuda.is_available():
        cuda_resutls = instance_data.cuda()
        for k, v in cuda_resutls.items():
            if isinstance(v, torch.Tensor):
                assert v.is_cuda


def test_instance_data():
    meta_info = dict(
        img_size=(256, 256),
        path='dadfaff',
        scale_factor=np.array([1.5, 1.5, 1, 1]))

    data = dict(
        bboxes=torch.rand(4, 4),
        masks=torch.rand(4, 2, 2),
        labels=np.random.rand(4),
        size=[(i, i) for i in range(4)])

    # test init
    instance_data = InstanceData(meta_info)
    assert 'path' in instance_data
    instance_data = InstanceData(meta_info, data=data)
    assert len(instance_data) == 4
    instance_data.set_data(data)
    assert len(instance_data) == 4

    meta_info = copy.deepcopy(meta_info)
    meta_info['img_name'] = 'flag'

    # test newinstance_data
    new_instance_data = instance_data.new(meta_info=meta_info)
    for k, v in new_instance_data.meta_info_items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert _equal(v, meta_info[k])
            assert k == 'img_name'
    # meta info is immutable
    with pytest.raises(KeyError):
        meta_info = copy.deepcopy(meta_info)
        meta_info['path'] = 'fdasfdsd'
        instance_data.new(meta_info=meta_info)

    # data fields should have same length
    with pytest.raises(AssertionError):
        temp_data = copy.deepcopy(data)
        temp_data['bboxes'] = torch.rand(5, 4)
        instance_data.new(data=temp_data)

    temp_data = copy.deepcopy(data)
    temp_data['scores'] = torch.rand(4)
    new_instance_data = instance_data.new(data=temp_data)
    for k, v in new_instance_data.items():
        if k in instance_data:
            _equal(v, instance_data[k])
        else:
            assert k == 'scores'
            assert _equal(v, temp_data[k])

    instance_data = instance_data.new()

    # test __setattr__
    # '_meta_info_field', '_data_fields' is immutable.
    with pytest.raises(AttributeError):
        instance_data._data_fields = dict()
    with pytest.raises(AttributeError):
        instance_data._data_fields = dict()

    # all attribute in instance_data_field should be
    # (torch.Tensor, np.ndarray, list))
    with pytest.raises(AssertionError):
        instance_data.a = 1000

    # instance_data field should has same length
    new_instance_data = instance_data.new()
    new_instance_data.det_bbox = torch.rand(100, 4)
    new_instance_data.det_label = torch.arange(100)
    with pytest.raises(AssertionError):
        new_instance_data.scores = torch.rand(101, 1)
    new_instance_data.none = [None] * 100
    with pytest.raises(AssertionError):
        new_instance_data.scores = [None] * 101
    new_instance_data.numpy_det = np.random.random([100, 1])
    with pytest.raises(AssertionError):
        new_instance_data.scores = np.random.random([101, 1])

    # isinstance(str, slice, int, torch.LongTensor, torch.BoolTensor)
    item = torch.Tensor([1, 2, 3, 4])
    with pytest.raises(AssertionError):
        new_instance_data[item]
    len(new_instance_data[item.long()]) == 1

    # when input is a bool tensor, The shape of
    # the input at index 0 should equal to
    # the value length in instance_data_field
    with pytest.raises(AssertionError):
        new_instance_data[item.bool()]

    for i in range(len(new_instance_data)):
        assert new_instance_data[i].det_label == i
        assert len(new_instance_data[i]) == 1

    # assert the index should in 0 ~ len(instance_data) -1
    with pytest.raises(IndexError):
        new_instance_data[101]

    # assert the index should not be an empty tensor
    new_new_instance_data = new_instance_data.new()
    with pytest.raises(AssertionError):
        new_new_instance_data[0]

    # test str
    with pytest.raises(AssertionError):
        instance_data.img_size_dummmy = meta_info['img_size']

    # test slice
    ten_ressults = new_instance_data[:10]
    len(ten_ressults) == 10
    for v in ten_ressults.values():
        assert len(v) == 10

    # test Longtensor
    long_tensor = torch.randint(100, (50, ))
    long_index_instance_data = new_instance_data[long_tensor]
    assert len(long_index_instance_data) == len(long_tensor)
    for key, value in long_index_instance_data.items():
        if not isinstance(value, list):
            assert (long_index_instance_data[key] == new_instance_data[key]
                    [long_tensor]).all()
        else:
            len(long_tensor) == len(value)

    # test bool tensor
    bool_tensor = torch.rand(100) > 0.5
    bool_index_instance_data = new_instance_data[bool_tensor]
    assert len(bool_index_instance_data) == bool_tensor.sum()
    for key, value in bool_index_instance_data.items():
        if not isinstance(value, list):
            assert (bool_index_instance_data[key] == new_instance_data[key]
                    [bool_tensor]).all()
        else:
            assert len(value) == bool_tensor.sum()

    num_instance = 1000
    instance_data_list = []

    # assert len(instance_lists) > 0
    with pytest.raises(AssertionError):
        instance_data.cat(instance_data_list)

    for _ in range(2):
        instance_data['bbox'] = torch.rand(num_instance, 4)
        instance_data['label'] = torch.rand(num_instance, 1)
        instance_data['mask'] = torch.rand(num_instance, 224, 224)
        instance_data['instances_infos'] = [1] * num_instance
        instance_data['cpu_bbox'] = np.random.random((num_instance, 4))
        if torch.cuda.is_available():
            instance_data.cuda_tensor = torch.rand(num_instance).cuda()
            assert instance_data.cuda_tensor.is_cuda
            cuda_instance_data = instance_data.cuda()
            assert cuda_instance_data.cuda_tensor.is_cuda

        assert len(instance_data[0]) == 1
        with pytest.raises(IndexError):
            return instance_data[num_instance + 1]
        with pytest.raises(AssertionError):
            instance_data.centerness = torch.rand(num_instance + 1, 1)

        mask_tensor = torch.rand(num_instance) > 0.5
        length = mask_tensor.sum()
        assert len(instance_data[mask_tensor]) == length

        index_tensor = torch.LongTensor([1, 5, 8, 110, 399])
        length = len(index_tensor)

        assert len(instance_data[index_tensor]) == length

        instance_data_list.append(instance_data)

    cat_resutls = InstanceData.cat(instance_data_list)
    assert len(cat_resutls) == num_instance * 2

    instances = InstanceData(data=dict(bboxes=torch.rand(4, 4)))
    # cat only single instance
    assert len(InstanceData.cat([instances])) == 4
