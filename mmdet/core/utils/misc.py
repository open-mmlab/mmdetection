# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch
from six.moves import map, zip

from ..mask.structures import BitmapMasks, PolygonMasks


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def mask2ndarray(mask):
    """Convert Mask to ndarray..

    Args:
        mask (:obj:`BitmapMasks` or :obj:`PolygonMasks` or
        torch.Tensor or np.ndarray): The mask to be converted.

    Returns:
        np.ndarray: Ndarray mask of shape (n, h, w) that has been converted
    """
    if isinstance(mask, (BitmapMasks, PolygonMasks)):
        mask = mask.to_ndarray()
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        raise TypeError(f'Unsupported {type(mask)} data type')
    return mask


def flip_tensor(src_tensor, flip_direction):
    """flip tensor base on flip_direction.

    Args:
        src_tensor (Tensor): input feature map, shape (B, C, H, W).
        flip_direction (str): The flipping direction. Options are
          'horizontal', 'vertical', 'diagonal'.

    Returns:
        out_tensor (Tensor): Flipped tensor.
    """
    assert src_tensor.ndim == 4
    valid_directions = ['horizontal', 'vertical', 'diagonal']
    assert flip_direction in valid_directions
    if flip_direction == 'horizontal':
        out_tensor = torch.flip(src_tensor, [3])
    elif flip_direction == 'vertical':
        out_tensor = torch.flip(src_tensor, [2])
    else:
        out_tensor = torch.flip(src_tensor, [2, 3])
    return out_tensor


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    """Extract a multi-scale single image tensor from a multi-scale batch
    tensor based on batch index.

    Note: The default value of detach is True, because the proposal gradient
    needs to be detached during the training of the two-stage model. E.g
    Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): Batch tensor for all scale levels,
           each is a 4D-tensor.
        batch_id (int): Batch index.
        detach (bool): Whether detach gradient. Default True.

    Returns:
        list[Tensor]: Multi-scale single image tensor.
    """
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter results using score threshold and topk candidates.

    Args:
        scores (Tensor): The scores, shape (num_bboxes, K).
        score_thr (float): The score filter threshold.
        topk (int): The number of topk candidates.
        results (dict or list or Tensor, Optional): The results to
           which the filtering rule is to be applied. The shape
           of each item is (num_bboxes, N).

    Returns:
        tuple: Filtered results

            - scores (Tensor): The scores after being filtered, \
                shape (num_bboxes_filtered, ).
            - labels (Tensor): The class labels, shape \
                (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): The anchor indexes, shape \
                (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional): \
                The filtered results. The shape of each item is \
                (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # torch.sort is actually faster than .topk (at least on GPUs)
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(f'Only supports dict or list or Tensor, '
                                      f'but get {type(results)}.')
    return scores, labels, keep_idxs, filtered_results


def center_of_mass(mask, esp=1e-6):
    """Calculate the centroid coordinates of the mask.

    Args:
        mask (Tensor): The mask to be calculated, shape (h, w).
        esp (float): Avoid dividing by zero. Default: 1e-6.

    Returns:
        tuple[Tensor]: the coordinates of the center point of the mask.

            - center_h (Tensor): the center point of the height.
            - center_w (Tensor): the center point of the width.
    """
    h, w = mask.shape
    grid_h = torch.arange(h, device=mask.device)[:, None]
    grid_w = torch.arange(w, device=mask.device)
    normalizer = mask.sum().float().clamp(min=esp)
    center_h = (mask * grid_h).sum() / normalizer
    center_w = (mask * grid_w).sum() / normalizer
    return center_h, center_w


def generate_coordinate(featmap_sizes, device='cuda'):
    """Generate the coordinate.

    Args:
        featmap_sizes (tuple): The feature to be calculated,
            of shape (N, C, W, H).
        device (str): The device where the feature will be put on.
    Returns:
        coord_feat (Tensor): The coordinate feature, of shape (N, 2, W, H).
    """

    x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
    y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([featmap_sizes[0], 1, -1, -1])
    x = x.expand([featmap_sizes[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)

    return coord_feat
