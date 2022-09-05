# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
import torch
from six.moves import map, zip

from ..mask.structures import BitmapMasks, PolygonMasks


def multi_apply(func, *args, **kwargs):
    """将函数func应用于参数列表args.
    设 def func(x):
           return str(x), x
    举个简单的例子.args为([3,10,5]),先忽略kwargs参数
    map_results = map(func, *args)
    那么list(map_results)就是[('3', 3), ('10', 10), ('5', 5)].
    那么zip(*map_results))就是[('3', '10', '5'), (3, 10, 5)]
    那么tuple(map(list, zip(*map_results)))就为(['3', '10', '5'], [3, 10, 5])

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    # args一般是tuple(list[])这种结构
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """将数据集映射回原始数据集上去)"""
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
    """根据batch索引从多尺度张量[(N C H W)] * num_level中提取多尺度单张图像张量.

    注意: detach默认为 True, 因为在两阶段模型的训练过程中需要截断proposal的梯度.
        比如 Cascade Mask R-CNN.

    Args:
        mlvl_tensors (list[Tensor]): 所有层级的张量(N C H W) * num_level,
        batch_id (int): Batch 索引.
        detach (bool): 是否截断输入的梯度.

    Returns:
        list[Tensor]: 单张图像的多尺度张量.
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
    """使用score_thr和topk过滤results.

    Args:
        scores (Tensor): box cls_scores, shape (num_bboxes, K).
        score_thr (float): box cls_scores的过滤阈值.
        topk (int): 前k个结果.
        results (dict or list or Tensor, Optional): 要应用过滤规则的结果.
            每个元素的shape (num_bboxes, N).

    Returns:
        tuple: 过滤后的结果,该结果也是根据score排过序的

            - scores (Tensor): 过滤后的score, shape (num_bboxes_filtered, ).
            - labels (Tensor): 过滤后的label, shape (num_bboxes_filtered, ).
            - anchor_idxs (Tensor): anchor索引, shape  (num_bboxes_filtered, ).
            - filtered_results (dict or list or Tensor, Optional):
                过滤后的结果. 每个元素的shape (num_bboxes_filtered, N).
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    # 因为valid_mask为二维数据,所以valid_idxs的shape为(n,2)
    # 其中2的第一列为box维度上的索引,第二列为cls维度上的索引
    # 同时 n <= num_box*k
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # 实际上torch.sort要比.topk更快 (至少在GPU上是这样)
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
