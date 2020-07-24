import numpy as np
import torch

from mmdet.datasets.pipelines import KeypointsToHeatmaps
from mmdet.models.builder import build_head


def test_all_inner_keypoints():
    """Tests whether keypoints can be converted to heatmaps and
    to keypoints back again
    NOTE: Heatmap is smaller than original image, and keypoints
    might be slightly shifted afte decoding the heatmaps
    This unit test might be a little too strict
    """
    dividers = [4, 4]
    sigma = 1.5
    label_type = 'Gaussian'
    to_heatmap = KeypointsToHeatmaps(
        dividers=dividers, sigma=sigma, label_type=label_type)
    pad_shape = [256, 256]

    to_keypoints = build_head(
        dict(type='HeatmapDecodeOneKeypoint', upscale=dividers))

    kpts = []
    for x_pos in range(2, 250, 1):
        for y_pos in range(2, 250, 1):
            kpts.append([x_pos, y_pos, 2])
    kpts = np.array(kpts).reshape(1, -1, 3)
    results = {'gt_keypoints': kpts, 'pad_shape': pad_shape}
    heatmaps = to_heatmap(results)['heatmaps']
    heatmaps = torch.Tensor(heatmaps).unsqueeze(0)
    keypoints = to_keypoints(heatmaps)

    for gt, pred, heatmap in zip(kpts[0], keypoints[0], heatmaps[0]):
        gtx, gty, v = gt
        pdx, pdy, predicted = pred
        assert pdx == gtx
        assert pdy == gty


def _process_inner_keypoints(dividers, sigmas, label_type, pad_shapes):
    """Creates a heatmap for every keypoint and convert to keypoint back again
    Allows some errors since the heatmap is smaller than the original image.

    Args:
        dividers ([type]): [description]
        sigmas ([type]): [description]
        label_type ([type]): [description]
        pad_shapes ([type]): [description]
    """
    for sigma in sigmas:
        for pad_shape in pad_shapes:
            to_heatmap = KeypointsToHeatmaps(
                dividers=dividers, sigma=sigma, label_type=label_type)
            to_keypoints = build_head(
                dict(type='HeatmapDecodeOneKeypoint', upscale=dividers))

            kpts = []
            for x_pos in range(10, pad_shape[1] - 10, 1):
                for y_pos in range(10, pad_shape[0] - 10, 1):
                    kpts.append([x_pos, y_pos, 2])
            kpts = np.array(kpts).reshape(1, -1, 3)
            results = {'gt_keypoints': kpts, 'pad_shape': pad_shape}
            heatmaps = to_heatmap(results)['heatmaps']
            heatmaps = torch.Tensor(heatmaps).unsqueeze(0)
            keypoints = to_keypoints(heatmaps)

            for gt, pred in zip(kpts[0], keypoints[0]):
                gtx, gty, v = gt
                pdx, pdy, predicted = pred
                assert np.abs(pdx - gtx) <= dividers[0]
                assert np.abs(pdy - gty) <= dividers[1]


def test_large_heatmaps():
    dividers = [4, 4]
    sigmas = [3]
    label_type = 'Gaussian'
    pad_shapes = [[384, 384]]
    _process_inner_keypoints(dividers, sigmas, label_type, pad_shapes)


def test_sigmas():
    dividers = [4, 4]
    sigmas = [1.5, 3, 4.5]
    label_type = 'Gaussian'
    pad_shapes = [[256, 256]]
    _process_inner_keypoints(dividers, sigmas, label_type, pad_shapes)


def test_boundary_keypoints():
    dividers = [4, 4]
    sigma = 1.5
    label_type = 'Gaussian'
    to_heatmap = KeypointsToHeatmaps(
        dividers=dividers, sigma=sigma, label_type=label_type)
    pad_shape = [256, 256]

    to_keypoints = build_head(
        dict(type='HeatmapDecodeOneKeypoint', upscale=dividers))

    kpts = []
    for y_pos in range(dividers[1], 250, dividers[1]):
        kpts.append([249, y_pos, 2])
    kpts = np.array(kpts).reshape(1, -1, 3)
    results = {'gt_keypoints': kpts, 'pad_shape': pad_shape}
    heatmaps = to_heatmap(results)['heatmaps']
    heatmaps = torch.Tensor(heatmaps).unsqueeze(0)
    keypoints = to_keypoints(heatmaps)

    for gt, pred in zip(kpts[0], keypoints[0]):
        gtx, gty, v = gt
        pdx, pdy, predicted = pred
        assert pdx == gtx
        assert pdy == gty
