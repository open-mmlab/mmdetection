# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet.structures.bbox import BaseBoxes


def find_inside_bboxes(bboxes: Tensor, img_h: int, img_w: int) -> Tensor:
    """Find bboxes as long as a part of bboxes is inside the image.

    Args:
        bboxes (Tensor): Shape (N, 4).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        Tensor: Index of the remaining bboxes.
    """
    inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) \
        & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
    return inside_inds


def bbox_flip(bboxes: Tensor,
              img_shape: Tuple[int],
              direction: str = 'horizontal') -> Tensor:
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (Tuple[int]): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"

    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_mapping(bboxes: Tensor,
                 img_shape: Tuple[int],
                 scale_factor: Union[float, Tuple[float]],
                 flip: bool,
                 flip_direction: str = 'horizontal') -> Tensor:
    """Map bboxes from the original image scale to testing scale."""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


def bbox_mapping_back(bboxes: Tensor,
                      img_shape: Tuple[int],
                      scale_factor: Union[float, Tuple[float]],
                      flip: bool,
                      flip_direction: str = 'horizontal') -> Tensor:
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def bbox2roi(bbox_list: List[Union[Tensor, BaseBoxes]]) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
        different box types. For example, If the box type in ``bbox_list``
        is HorizontalBoxes, the output shape is (n, 5). Each row of data
        indicates [batch_ind, x1, y1, x2, y2].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        bboxes = get_box_tensor(bboxes)
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois: Tensor) -> List[Tensor]:
    """Convert rois to bounding box format.

    Args:
        rois (Tensor): RoIs with the shape (n, 5) where the first
            column indicates batch id of each RoI.

    Returns:
        List[Tensor]: Converted boxes of corresponding rois.
    """
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


# TODO remove later
def bbox2result(bboxes: Union[Tensor, np.ndarray], labels: Union[Tensor,
                                                                 np.ndarray],
                num_classes: int) -> List[np.ndarray]:
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor | np.ndarray): shape (n, 5)
        labels (Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        List(np.ndarray]): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape: Optional[Union[Sequence[int], Tensor,
                              Sequence[Sequence[int]]]] = None
) -> Tensor:
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (B, N, 2) or (N, 2).
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)
        max_shape (Union[Sequence[int], Tensor, Sequence[Sequence[int]]],
            optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.

    Returns:
        Tensor: Boxes with shape (N, 4) or (B, N, 4)
    """

    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            # speed up
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes

        # clip bboxes with dynamic `min` and `max` for onnx
        if torch.onnx.is_in_onnx_export():
            # TODO: delete
            from mmdet.core.export import dynamic_clip_for_onnx
            x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, max_shape)
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


def bbox2distance(points: Tensor,
                  bbox: Tensor,
                  max_dis: Optional[float] = None,
                  eps: float = 0.1) -> Tensor:
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2) or (b, n, 2), [x, y].
        bbox (Tensor): Shape (n, 4) or (b, n, 4), "xyxy" format
        max_dis (float, optional): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def bbox_rescale(bboxes: Tensor, scale_factor: float = 1.0) -> Tensor:
    """Rescale bounding box w.r.t. scale_factor.

    Args:
        bboxes (Tensor): Shape (n, 4) for bboxes or (n, 5) for rois
        scale_factor (float): rescale factor

    Returns:
        Tensor: Rescaled bboxes.
    """
    if bboxes.size(1) == 5:
        bboxes_ = bboxes[:, 1:]
        inds_ = bboxes[:, 0]
    else:
        bboxes_ = bboxes
    cx = (bboxes_[:, 0] + bboxes_[:, 2]) * 0.5
    cy = (bboxes_[:, 1] + bboxes_[:, 3]) * 0.5
    w = bboxes_[:, 2] - bboxes_[:, 0]
    h = bboxes_[:, 3] - bboxes_[:, 1]
    w = w * scale_factor
    h = h * scale_factor
    x1 = cx - 0.5 * w
    x2 = cx + 0.5 * w
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    if bboxes.size(1) == 5:
        rescaled_bboxes = torch.stack([inds_, x1, y1, x2, y2], dim=-1)
    else:
        rescaled_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return rescaled_bboxes


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def bbox2corner(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to corners ((x1, y1),
    (x2, y1), (x1, y2), (x2, y2)).

    Args:
        bboxes (Tensor): Shape (n, 4) for bboxes.
    Returns:
        Tensor: Shape (n*4, 2) for corners.
    """
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=1)
    return torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=1).reshape(-1, 2)


def corner2bbox(corners: torch.Tensor) -> torch.Tensor:
    """Convert bbox coordinates from corners ((x1, y1), (x2, y1), (x1, y2),
    (x2, y2)) to (x1, y1, x2, y2).

    Args:
        corners (Tensor): Shape (n*4, 2) for corners.
    Returns:
        Tensor: Shape (n, 4) for bboxes.
    """
    corners = corners.reshape(-1, 4, 2)
    min_xy = corners.min(dim=1)[0]
    max_xy = corners.max(dim=1)[0]
    return torch.cat([min_xy, max_xy], dim=1)


def bbox_project(
    bboxes: Union[torch.Tensor, np.ndarray],
    homography_matrix: Union[torch.Tensor, np.ndarray],
    img_shape: Optional[Tuple[int, int]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """Geometric transformation for bbox.

    Args:
        bboxes (Union[torch.Tensor, np.ndarray]): Shape (n, 4) for bboxes.
        homography_matrix (Union[torch.Tensor, np.ndarray]):
            Shape (3, 3) for geometric transformation.
        img_shape (Tuple[int, int], optional): Image shape. Defaults to None.
    Returns:
        Union[torch.Tensor, np.ndarray]: Converted bboxes.
    """
    bboxes_type = type(bboxes)
    if bboxes_type is np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if isinstance(homography_matrix, np.ndarray):
        homography_matrix = torch.from_numpy(homography_matrix)
    corners = bbox2corner(bboxes)
    corners = torch.cat(
        [corners, corners.new_ones(corners.shape[0], 1)], dim=1)
    corners = torch.matmul(homography_matrix, corners.t()).t()
    # Convert to homogeneous coordinates by normalization
    corners = corners[:, :2] / corners[:, 2:3]
    bboxes = corner2bbox(corners)
    if img_shape is not None:
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, img_shape[1])
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, img_shape[0])
    if bboxes_type is np.ndarray:
        bboxes = bboxes.numpy()
    return bboxes


def cat_boxes(data_list: List[Union[Tensor, BaseBoxes]],
              dim: int = 0) -> Union[Tensor, BaseBoxes]:
    """Concatenate boxes with type of tensor or BoxList.

    Args:
        data_list (List[Union[Tensor, :obj:`BaseBoxes`]]): A list of tensors
            or boxlists need to be concatenated.
            dim (int): The dimension over which the box are concatenated.
                Defaults to 0.

    Returns:
        Union[Tensor, :obj`BaseBoxes`]: Concatenated results.
    """
    if data_list and isinstance(data_list[0], BaseBoxes):
        return data_list[0].cat(data_list, dim=dim)
    else:
        return torch.cat(data_list, dim=dim)


def stack_boxes(data_list: List[Union[Tensor, BaseBoxes]],
                dim: int = 0) -> Union[Tensor, BaseBoxes]:
    """Stack boxes with type of tensor or BoxList.

    Args:
        data_list (List[Union[Tensor, :obj:`BaseBoxes`]]): A list of tensors
            or boxlists need to be stacked.
            dim (int): The dimension over which the box are stacked.
                Defaults to 0.

    Returns:
        Union[Tensor, :obj`BaseBoxes`]: Stacked results.
    """
    if data_list and isinstance(data_list[0], BaseBoxes):
        return data_list[0].stack(data_list, dim=dim)
    else:
        return torch.stack(data_list, dim=dim)


def scale_boxes(boxes: Union[Tensor, BaseBoxes],
                scale_factor: Tuple[float, float]) -> Union[Tensor, BaseBoxes]:
    """Scale boxes with type of tensor or boxlist.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes need to be scaled. Its type
            can be a tensor or a boxlist.
        scale_factor (Tuple[float, float]): factors for scaling boxes.
            The length should be 2.

    Returns:
        Union[Tensor, :obj:`BaseBoxes`]: Scaled boxes.
    """
    if isinstance(boxes, BaseBoxes):
        boxes.rescale_(scale_factor)
        return boxes
    else:
        # Tensor boxes will be treated as horizontal boxes
        repeat_num = int(boxes.size(-1) / 2)
        scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
        return boxes * scale_factor


def get_box_wh(boxes: Union[Tensor, BaseBoxes]) -> Tuple[Tensor, Tensor]:
    """Get the width and height of boxes with type of tensor or boxlist.

    Args:
        boxes (Tensor or :obj:`BaseBoxes`): boxes with type of tensor
            or boxlist.

    Returns:
        Tuple[Tensor, Tensor]: the width and height of boxes.
    """
    if isinstance(boxes, BaseBoxes):
        w = boxes.widths
        h = boxes.heights
    else:
        # Tensor boxes will be treated as horizontal boxes by defaults
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
    return w, h


def get_box_tensor(boxes: Union[Tensor, BaseBoxes]) -> Tensor:
    """Get tensor data from boxlist boxes.

    Args:
        boxes (Tensor or BaseBoxes): boxes with type of tensor or boxlist.
            If its type is a tensor, the boxes will be directly returned.
            If its type is a boxlist, the `boxes.tensor` will be returned.

    Returns:
        Tensor: boxes tensor.
    """
    if isinstance(boxes, BaseBoxes):
        boxes = boxes.tensor
    return boxes
