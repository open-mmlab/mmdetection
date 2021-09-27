import pytest
import torch

from mmdet.core.post_processing import mask_matrix_nms


def _create_mask(N, h, w):
    masks = torch.rand((N, h, w)) > 0.5
    labels = torch.rand(N)
    scores = torch.rand(N)
    return masks, labels, scores


def test_nms_input_errors():
    with pytest.raises(AssertionError):
        mask_matrix_nms(
            torch.rand((10, 28, 28)), torch.rand(11), torch.rand(11))
    with pytest.raises(AssertionError):
        masks = torch.rand((10, 28, 28))
        mask_matrix_nms(
            masks,
            torch.rand(11),
            torch.rand(11),
            mask_area=masks.sum((1, 2)).float()[:8])
    with pytest.raises(NotImplementedError):
        mask_matrix_nms(
            torch.rand((10, 28, 28)),
            torch.rand(10),
            torch.rand(10),
            kernel='None')
    # test an empty results
    masks, labels, scores = _create_mask(0, 28, 28)
    score, label, mask, keep_ind = \
        mask_matrix_nms(masks, labels, scores)
    assert len(score) == len(label) == \
           len(mask) == len(keep_ind) == 0

    # do not use update_thr, nms_pre and max_num
    masks, labels, scores = _create_mask(1000, 28, 28)
    score, label, mask, keep_ind = \
        mask_matrix_nms(masks, labels, scores)
    assert len(score) == len(label) == \
           len(mask) == len(keep_ind) == 1000
    # only use nms_pre
    score, label, mask, keep_ind = \
        mask_matrix_nms(masks, labels, scores, nms_pre=500)
    assert len(score) == len(label) == \
           len(mask) == len(keep_ind) == 500
    # use max_num
    score, label, mask, keep_ind = \
        mask_matrix_nms(masks, labels, scores,
                        nms_pre=500, max_num=100)
    assert len(score) == len(label) == \
           len(mask) == len(keep_ind) == 100

    masks, labels, _ = _create_mask(1, 28, 28)
    scores = torch.Tensor([1.0])
    masks = masks.expand(1000, 28, 28)
    labels = labels.expand(1000)
    scores = scores.expand(1000)

    # assert scores is decayed and update_thr is worked
    # if with the same mask, label, and all scores = 1
    # the first score will set to 1, others will decay.
    score, label, mask, keep_ind = \
        mask_matrix_nms(masks,
                        labels,
                        scores,
                        nms_pre=500,
                        max_num=100,
                        kernel='gaussian',
                        sigma=2.0,
                        filter_thr=0.5)
    assert len(score) == 1
    assert score[0] == 1
