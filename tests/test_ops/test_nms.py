"""
CommandLine:
    pytest tests/test_nms.py
"""
import numpy as np
import pytest
import torch

from mmdet.ops.nms.nms_wrapper import nms, nms_match


def test_nms_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_cpu
    """
    iou_thr = 0.6
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.1],
                          [49.3, 32.9, 51.0, 35.3, 0.05],
                          [35.3, 11.5, 39.9, 14.5, 0.9],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    base_expected_suppressed = np.array([[35.3, 11.5, 39.9, 14.5, 0.9],
                                         [49.1, 32.4, 51.0, 35.9, 0.1]])
    # CPU can handle float32 and float64
    dets = base_dets.astype(np.float32)
    expected_suppressed = base_expected_suppressed.astype(np.float32)
    suppressed, inds = nms(dets, iou_thr)
    assert dets.dtype == suppressed.dtype
    assert np.array_equal(suppressed, expected_suppressed)

    dets = torch.FloatTensor(base_dets)
    expected_suppressed = torch.FloatTensor(base_expected_suppressed)
    suppressed, inds = nms(dets, iou_thr)
    assert dets.dtype == suppressed.dtype
    assert torch.equal(suppressed, expected_suppressed)

    dets = base_dets.astype(np.float64)
    expected_suppressed = base_expected_suppressed.astype(np.float64)
    suppressed, inds = nms(dets, iou_thr)
    assert dets.dtype == suppressed.dtype
    assert np.array_equal(suppressed, expected_suppressed)

    dets = torch.DoubleTensor(base_dets)
    expected_suppressed = torch.DoubleTensor(base_expected_suppressed)
    suppressed, inds = nms(dets, iou_thr)
    assert dets.dtype == suppressed.dtype
    assert torch.equal(suppressed, expected_suppressed)


def test_nms_device_and_dtypes_gpu():
    """
    CommandLine:
        xdoctest -m tests/test_nms.py test_nms_device_and_dtypes_gpu
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('test requires GPU and torch+cuda')

    iou_thr = 0.6
    base_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.1],
                          [49.3, 32.9, 51.0, 35.3, 0.05],
                          [35.3, 11.5, 39.9, 14.5, 0.9],
                          [35.2, 11.7, 39.7, 15.7, 0.3]])

    base_expected_suppressed = np.array([[35.3, 11.5, 39.9, 14.5, 0.9],
                                         [49.1, 32.4, 51.0, 35.9, 0.1]])

    for device_id in range(torch.cuda.device_count()):
        print(f'Run NMS on device_id = {device_id!r}')
        # GPU can handle float32 but not float64
        dets = base_dets.astype(np.float32)
        expected_suppressed = base_expected_suppressed.astype(np.float32)
        suppressed, inds = nms(dets, iou_thr, device_id)
        assert dets.dtype == suppressed.dtype
        assert np.array_equal(suppressed, expected_suppressed)

        dets = torch.FloatTensor(base_dets).to(device_id)
        expected_suppressed = torch.FloatTensor(base_expected_suppressed).to(
            device_id)
        suppressed, inds = nms(dets, iou_thr)
        assert dets.dtype == suppressed.dtype
        assert torch.equal(suppressed, expected_suppressed)


def test_nms_match():
    iou_thr = 0.6
    # empty input
    empty_dets = np.array([])
    assert len(nms_match(empty_dets, iou_thr)) == 0

    # non empty ndarray input
    np_dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
                        [49.3, 32.9, 51.0, 35.3, 0.9],
                        [35.3, 11.5, 39.9, 14.5, 0.4],
                        [35.2, 11.7, 39.7, 15.7, 0.3]])
    np_groups = nms_match(np_dets, iou_thr)
    assert isinstance(np_groups[0], np.ndarray)
    assert len(np_groups) == 2
    nms_keep_inds = nms(np_dets, iou_thr)[1]
    assert set([g[0].item() for g in np_groups]) == set(nms_keep_inds.tolist())

    # non empty tensor input
    tensor_dets = torch.from_numpy(np_dets)
    tensor_groups = nms_match(tensor_dets, iou_thr)
    assert isinstance(tensor_groups[0], torch.Tensor)
    for i in range(len(tensor_groups)):
        assert np.equal(tensor_groups[i].numpy(), np_groups[i]).all()

    # input of wrong shape
    wrong_dets = np.zeros((2, 3))
    with pytest.raises(AssertionError):
        nms_match(wrong_dets, iou_thr)
