# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import digit_version
from six.moves import map, zip
from torch import Tensor
from torch.autograd import Function
from torch.nn import functional as F

from mmdet.structures import SampleList
from mmdet.structures.bbox import BaseBoxes, get_box_type, stack_boxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import OptInstanceList


class SigmoidGeometricMean(Function):
    """Forward and backward function of geometric mean of two sigmoid
    functions.

    This implementation with analytical gradient function substitutes
    the autograd function of (x.sigmoid() * y.sigmoid()).sqrt(). The
    original implementation incurs none during gradient backprapagation
    if both x and y are very small values.
    """

    @staticmethod
    def forward(ctx, x, y):
        x_sigmoid = x.sigmoid()
        y_sigmoid = y.sigmoid()
        z = (x_sigmoid * y_sigmoid).sqrt()
        ctx.save_for_backward(x_sigmoid, y_sigmoid, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x_sigmoid, y_sigmoid, z = ctx.saved_tensors
        grad_x = grad_output * z * (1 - x_sigmoid) / 2
        grad_y = grad_output * z * (1 - y_sigmoid) / 2
        return grad_x, grad_y


sigmoid_geometric_mean = SigmoidGeometricMean.apply


def interpolate_as(source, target, mode='bilinear', align_corners=False):
    """Interpolate the `source` to the shape of the `target`.

    The `source` must be a Tensor, but the `target` can be a Tensor or a
    np.ndarray with the shape (..., target_h, target_w).

    Args:
        source (Tensor): A 3D/4D Tensor with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The interpolation target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for interpolation. The options are the
            same as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The interpolated source Tensor.
    """
    assert len(target.shape) >= 2

    def _interpolate_as(source, target, mode='bilinear', align_corners=False):
        """Interpolate the `source` (4D) to the shape of the `target`."""
        target_h, target_w = target.shape[-2:]
        source_h, source_w = source.shape[-2:]
        if target_h != source_h or target_w != source_w:
            source = F.interpolate(
                source,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return source

    if len(source.shape) == 3:
        source = source[:, None, :, :]
        source = _interpolate_as(source, target, mode, align_corners)
        return source[:, 0, :, :]
    else:
        return _interpolate_as(source, target, mode, align_corners)


def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def empty_instances(batch_img_metas: List[dict],
                    device: torch.device,
                    task_type: str,
                    instance_results: OptInstanceList = None,
                    mask_thr_binary: Union[int, float] = 0,
                    box_type: Union[str, type] = 'hbox',
                    use_box_type: bool = False,
                    num_classes: int = 80,
                    score_per_cls: bool = False) -> List[InstanceData]:
    """Handle predicted instances when RoI is empty.

    Note: If ``instance_results`` is not None, it will be modified
    in place internally, and then return ``instance_results``

    Args:
        batch_img_metas (list[dict]): List of image information.
        device (torch.device): Device of tensor.
        task_type (str): Expected returned task type. it currently
            supports bbox and mask.
        instance_results (list[:obj:`InstanceData`]): List of instance
            results.
        mask_thr_binary (int, float): mask binarization threshold.
            Defaults to 0.
        box_type (str or type): The empty box type. Defaults to `hbox`.
        use_box_type (bool): Whether to warp boxes with the box type.
            Defaults to False.
        num_classes (int): num_classes of bbox_head. Defaults to 80.
        score_per_cls (bool):  Whether to generate classwise score for
            the empty instance. ``score_per_cls`` will be True when the model
            needs to produce raw results without nms. Defaults to False.

    Returns:
        list[:obj:`InstanceData`]: Detection results of each image
    """
    assert task_type in ('bbox', 'mask'), 'Only support bbox and mask,' \
                                          f' but got {task_type}'

    if instance_results is not None:
        assert len(instance_results) == len(batch_img_metas)

    results_list = []
    for img_id in range(len(batch_img_metas)):
        if instance_results is not None:
            results = instance_results[img_id]
            assert isinstance(results, InstanceData)
        else:
            results = InstanceData()

        if task_type == 'bbox':
            _, box_type = get_box_type(box_type)
            bboxes = torch.zeros(0, box_type.box_dim, device=device)
            if use_box_type:
                bboxes = box_type(bboxes, clone=False)
            results.bboxes = bboxes
            score_shape = (0, num_classes + 1) if score_per_cls else (0, )
            results.scores = torch.zeros(score_shape, device=device)
            results.labels = torch.zeros((0, ),
                                         device=device,
                                         dtype=torch.long)
        else:
            # TODO: Handle the case where rescale is false
            img_h, img_w = batch_img_metas[img_id]['ori_shape'][:2]
            # the type of `im_mask` will be torch.bool or torch.uint8,
            # where uint8 if for visualization and debugging.
            im_mask = torch.zeros(
                0,
                img_h,
                img_w,
                device=device,
                dtype=torch.bool if mask_thr_binary >= 0 else torch.uint8)
            results.masks = im_mask
        results_list.append(results)
    return results_list


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


def levels_to_images(mlvl_tensor: List[torch.Tensor]) -> List[torch.Tensor]:
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.

    Args:
        mlvl_tensor (list[Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)

    Returns:
        list[Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = stack_boxes(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def samplelist_boxtype2tensor(batch_data_samples: SampleList) -> SampleList:
    for data_samples in batch_data_samples:
        if 'gt_instances' in data_samples:
            bboxes = data_samples.gt_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if 'pred_instances' in data_samples:
            bboxes = data_samples.pred_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if 'ignored_instances' in data_samples:
            bboxes = data_samples.ignored_instances.get('bboxes', None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor


_torch_version_div_indexing = (
    'parrots' not in torch.__version__
    and digit_version(torch.__version__) >= digit_version('1.8'))


def floordiv(dividend, divisor, rounding_mode='trunc'):
    if _torch_version_div_indexing:
        return torch.div(dividend, divisor, rounding_mode=rounding_mode)
    else:
        return dividend // divisor


def _filter_gt_instances_by_score(batch_data_samples: SampleList,
                                  score_thr: float) -> SampleList:
    """Filter ground truth (GT) instances by score.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        assert 'scores' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.scores > score_thr]
    return batch_data_samples


def _filter_gt_instances_by_size(batch_data_samples: SampleList,
                                 wh_thr: tuple) -> SampleList:
    """Filter ground truth (GT) instances by size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        bboxes = data_samples.gt_instances.bboxes
        if bboxes.shape[0] > 0:
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            data_samples.gt_instances = data_samples.gt_instances[
                (w > wh_thr[0]) & (h > wh_thr[1])]
    return batch_data_samples


def filter_gt_instances(batch_data_samples: SampleList,
                        score_thr: float = None,
                        wh_thr: tuple = None):
    """Filter ground truth (GT) instances by score and/or size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score and/or size.
    """

    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score(
            batch_data_samples, score_thr)
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr)
    return batch_data_samples


def rename_loss_dict(prefix: str, losses: dict) -> dict:
    """Rename the key names in loss dict by adding a prefix.

    Args:
        prefix (str): The prefix for loss components.
        losses (dict):  A dictionary of loss components.

    Returns:
            dict: A dictionary of loss components with prefix.
    """
    return {prefix + k: v for k, v in losses.items()}


def reweight_loss_dict(losses: dict, weight: float) -> dict:
    """Reweight losses in the dict by weight.

    Args:
        losses (dict):  A dictionary of loss components.
        weight (float): Weight for loss components.

    Returns:
            dict: A dictionary of weighted loss components.
    """
    for name, loss in losses.items():
        if 'loss' in name:
            if isinstance(loss, Sequence):
                losses[name] = [item * weight for item in loss]
            else:
                losses[name] = loss * weight
    return losses


def relative_coordinate_maps(
    locations: Tensor,
    centers: Tensor,
    strides: Tensor,
    size_of_interest: int,
    feat_sizes: Tuple[int],
) -> Tensor:
    """Generate the relative coordinate maps with feat_stride.

    Args:
        locations (Tensor): The prior location of mask feature map.
            It has shape (num_priors, 2).
        centers (Tensor): The prior points of a object in
            all feature pyramid. It has shape (num_pos, 2)
        strides (Tensor): The prior strides of a object in
            all feature pyramid. It has shape (num_pos, 1)
        size_of_interest (int): The size of the region used in rel coord.
        feat_sizes (Tuple[int]): The feature size H and W, which has 2 dims.
    Returns:
        rel_coord_feat (Tensor): The coordinate feature
            of shape (num_pos, 2, H, W).
    """

    H, W = feat_sizes
    rel_coordinates = centers.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
    rel_coordinates = rel_coordinates.permute(0, 2, 1).float()
    rel_coordinates = rel_coordinates / (
        strides[:, None, None] * size_of_interest)
    return rel_coordinates.reshape(-1, 2, H, W)


def aligned_bilinear(tensor: Tensor, factor: int) -> Tensor:
    """aligned bilinear, used in original implement in CondInst:

    https://github.com/aim-uofa/AdelaiDet/blob/\
    c0b2092ce72442b0f40972f7c6dda8bb52c46d16/adet/utils/comm.py#L23
    """

    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0), mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


def unfold_wo_center(x, kernel_size: int, dilation: int) -> Tensor:
    """unfold_wo_center, used in original implement in BoxInst:

    https://github.com/aim-uofa/AdelaiDet/blob/\
    4a3a1f7372c35b48ebf5f6adc59f135a0fa28d60/\
    adet/modeling/condinst/condinst.py#L53
    """
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size, padding=padding, dilation=dilation)
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3))
    # remove the center pixels
    size = kernel_size**2
    unfolded_x = torch.cat(
        (unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]),
        dim=2)

    return unfolded_x


def padding_to(input_tensor: Tensor, max_len: int = 300) -> Tensor:
    """Pad the first dimension of `input_tensor` to `max_len`.

    Args:
        input_tensor (Tensor): The tensor to be padded,
        max_len (int): Padding target size in the first dimension.
            Default: 300
    https://github.com/jshilong/DDQ/blob/ddq_detr/projects/models/utils.py#L19
    Returns:
        Tensor: The tensor padded with the first dimension size `max_len`.
    """
    if max_len is None:
        return input_tensor
    num_padding = max_len - len(input_tensor)
    if input_tensor.dim() > 1:
        padding = input_tensor.new_zeros(
            num_padding, *input_tensor.size()[1:], dtype=input_tensor.dtype)
    else:
        padding = input_tensor.new_zeros(num_padding, dtype=input_tensor.dtype)
    output_tensor = torch.cat([input_tensor, padding], dim=0)
    return output_tensor


def align_tensor(inputs: List[Tensor],
                 max_len: Optional[int] = None) -> Tensor:
    """Pad each input to `max_len`, then stack them. If `max_len` is None, then
    it is the max size of the first dimension of each input.

        https://github.com/jshilong/DDQ/blob/ddq_detr/projects/models/\
        utils.py#L12

    Args:
        inputs (list[Tensor]): The tensors to be padded,
            Each input should have the same shape except the first dimension.
        max_len (int): Padding target size in the first dimension.
            Default: None
    Returns:
        Tensor: Stacked inputs after padding in the first dimension.
    """
    if max_len is None:
        max_len = max([len(item) for item in inputs])

    return torch.stack([padding_to(item, max_len) for item in inputs])
