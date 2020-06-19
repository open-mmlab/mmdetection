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
    base_bboxes = np.array([[49.1, 32.4, 51.0, 35.9], [49.3, 32.9, 51.0, 35.3],
                            [35.3, 11.5, 39.9, 14.5], [35.2, 11.7, 39.7,
                                                       15.7]])
    base_scores = np.array([0.1, 0.05, 0.9, 0.3])

    base_expected_suppressed = np.array([[35.3, 11.5, 39.9, 14.5, 0.9],
                                         [49.1, 32.4, 51.0, 35.9, 0.1]])
    # CPU can handle float32 and float64
    bboxes = base_bboxes.astype(np.float32)
    scores = base_scores.astype(np.float32)
    expected_suppressed = base_expected_suppressed.astype(np.float32)
    suppressed, inds = nms(bboxes, scores, iou_thr)
    assert bboxes.dtype == suppressed.dtype
    assert np.array_equal(suppressed, expected_suppressed)

    bboxes = torch.FloatTensor(base_bboxes)
    scores = torch.FloatTensor(base_scores)
    expected_suppressed = torch.FloatTensor(base_expected_suppressed)
    suppressed, inds = nms(bboxes, scores, iou_thr)
    assert bboxes.dtype == suppressed.dtype
    assert torch.equal(suppressed, expected_suppressed)

    bboxes = base_bboxes.astype(np.float64)
    scores = base_scores.astype(np.float64)
    expected_suppressed = base_expected_suppressed.astype(np.float64)
    suppressed, inds = nms(bboxes, scores, iou_thr)
    assert bboxes.dtype == suppressed.dtype
    assert np.array_equal(suppressed, expected_suppressed)

    bboxes = torch.DoubleTensor(base_bboxes)
    scores = torch.DoubleTensor(base_scores)
    expected_suppressed = torch.DoubleTensor(base_expected_suppressed)
    suppressed, inds = nms(bboxes, scores, iou_thr)
    assert bboxes.dtype == suppressed.dtype
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
    base_bboxes = np.array([[49.1, 32.4, 51.0, 35.9], [49.3, 32.9, 51.0, 35.3],
                            [35.3, 11.5, 39.9, 14.5], [35.2, 11.7, 39.7,
                                                       15.7]])
    base_scores = np.array([0.1, 0.05, 0.9, 0.3])

    base_expected_suppressed = np.array([[35.3, 11.5, 39.9, 14.5, 0.9],
                                         [49.1, 32.4, 51.0, 35.9, 0.1]])

    for device_id in range(torch.cuda.device_count()):
        print(f'Run NMS on device_id = {device_id!r}')
        # GPU can handle float32 but not float64
        bboxes = base_bboxes.astype(np.float32)
        scores = base_scores.astype(np.float32)
        expected_suppressed = base_expected_suppressed.astype(np.float32)
        suppressed, inds = nms(bboxes, scores, iou_thr, device_id)
        assert bboxes.dtype == suppressed.dtype
        assert np.array_equal(suppressed, expected_suppressed)

        bboxes = torch.FloatTensor(base_bboxes).to(device_id)
        scores = torch.FloatTensor(base_scores).to(device_id)
        expected_suppressed = torch.FloatTensor(base_expected_suppressed).to(
            device_id)
        suppressed, inds = nms(bboxes, scores, iou_thr)
        assert bboxes.dtype == suppressed.dtype
        assert torch.equal(suppressed, expected_suppressed)


def test_nms_match():
    iou_thr = 0.6
    # empty input
    empty_dets = np.array([])
    assert len(nms_match(empty_dets, iou_thr)) == 0

    # non empty ndarray input
    np_bboxes = np.array([[49.1, 32.4, 51.0, 35.9], [49.3, 32.9, 51.0, 35.3],
                          [35.3, 11.5, 39.9, 14.5], [35.2, 11.7, 39.7, 15.7]])
    np_scores = np.array([0.9, 0.9, 0.4, 0.3])
    np_dets = np.concatenate((np_bboxes, np_scores[:, None]), axis=-1)

    np_groups = nms_match(np_dets, iou_thr)
    assert isinstance(np_groups[0], np.ndarray)
    assert len(np_groups) == 2
    nms_keep_inds = nms(np_bboxes, np_scores, iou_thr)[1]
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
