import copy

import numpy as np
import pytest
import torch

from mmdet.core.results.results import Results


def test_results():
    meta_info = dict(
        img_size=(256, 256), path='dadfaff', scale_factor=np.array([1.5, 1.5]))
    results = Results(meta_info)
    results.a = 1000
    results['a'] = 2000
    assert results.a == 2000
    results._meta = 1000
    results.bbox = torch.ones(2, 3, 4, 5).cuda()
    results.bbox.sum()
    with pytest.raises(AttributeError):
        results.img_size = 100

    for k, v in results.items():
        if k == 'bbox':
            assert isinstance(v, torch.Tensor)
    assert results.has('a')
    results.remove('a')
    assert not results.has('a')

    new_results = results.cuda()
    new_results.bbox[0] = 100
    assert new_results.bbox.sum() != results.bbox.sum()
    cpu_results = new_results.cpu()
    for k, v in cpu_results.items():
        if isinstance(v, torch.Tensor):
            assert not v.is_cuda

    assert isinstance(cpu_results.numpy().bbox, np.ndarray)
    dump_results = results.export_results()
    assert 'img_size' not in dump_results
    copy_results = copy.copy(results)
    assert copy_results.bbox is results.bbox
    deep_copy = copy.deepcopy(results)
    assert deep_copy.bbox is not results.bbox
