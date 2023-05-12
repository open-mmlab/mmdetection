# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .bbox import bbox2corner, corner2bbox
from .mask import BitmapMasks


def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor."""
    if not isinstance(obj, torch.Tensor):
        raise TypeError('Input type is not a torch.Tensor. Got {}'.format(
            type(obj)))


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    tr_mat = torch.tensor(
        [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(
    dst_pix_trans_src_pix: torch.Tensor,
    dsize_src: Tuple[int, int],
    dsize_dst: Tuple[int, int],
) -> torch.Tensor:
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3
            or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            'Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}'.
            format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h, src_w).to(dst_pix_trans_src_pix)
    src_pix_trans_src_norm = torch.inverse(src_norm_trans_src_pix.float()).to(
        src_norm_trans_src_pix.dtype)
    dst_norm_trans_dst_pix: torch.Tensor = normal_transform_pixel(
        dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def warp_affine(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: Tuple[int, int],
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    if not isinstance(src, torch.Tensor):
        raise TypeError('Input src type is not a torch.Tensor. Got {}'.format(
            type(src)))

    if not isinstance(M, torch.Tensor):
        raise TypeError('Input M type is not a torch.Tensor. Got {}'.format(
            type(M)))

    if not len(src.shape) == 4:
        raise ValueError('Input src must be a BxCxHxW tensor. Got {}'.format(
            src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError('Input M must be a Bx2x3 tensor. Got {}'.format(
            M.shape))

    # TODO: remove the statement below in kornia v0.6
    if align_corners is None:
        message: str = (
            'The align_corners default value has been changed. By default now is set True '
            'in order to match cv2.warpAffine.')
        warnings.warn(message)
        # set default value for align corners
        align_corners = True

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine

    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(
        M, (H, W), dsize)

    src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm.float())

    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :],
        [B, C, dsize[0], dsize[1]],
        align_corners=align_corners,
    )
    return F.grid_sample(
        src.float(),
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode,
    ).to(src.dtype)


class Transform2D:

    @staticmethod
    def transform_bboxes(bbox, M, out_shape):
        if isinstance(bbox, Sequence):
            assert len(bbox) == len(M)
            return [
                Transform2D.transform_bboxes(b, m, o)
                for b, m, o in zip(bbox, M, out_shape)
            ]
        else:
            if bbox.shape[0] == 0:
                return bbox
            score = None
            if bbox.shape[1] > 4:
                score = bbox[:, 4:]
            points = bbox2corner(bbox[:, :4])
            points = torch.cat(
                [points, points.new_ones(points.shape[0], 1)], dim=1)  # n,3
            points = torch.matmul(M, points.t()).t()
            points = points[:, :2] / points[:, 2:3]
            bbox = corner2bbox(points, out_shape[1], out_shape[0])
            if score is not None:
                return torch.cat([bbox, score], dim=1)
            return bbox

    @staticmethod
    def transform_masks(
        mask: Union[BitmapMasks, List[BitmapMasks]],
        M: Union[torch.Tensor, List[torch.Tensor]],
        out_shape: Union[list, List[list]],
    ):
        if isinstance(mask, Sequence):
            assert len(mask) == len(M)
            return [
                Transform2D.transform_masks(b, m, o)
                for b, m, o in zip(mask, M, out_shape)
            ]
        else:
            if mask.masks.shape[0] == 0:
                return BitmapMasks(np.zeros((0, *out_shape)), *out_shape)
            mask_tensor = (
                torch.from_numpy(mask.masks[:, None,
                                            ...]).to(M.device).to(M.dtype))
            return BitmapMasks(
                warp_affine(
                    mask_tensor,
                    M[None, ...].expand(mask.masks.shape[0], -1, -1),
                    out_shape,
                ).squeeze(1).cpu().numpy(),
                out_shape[0],
                out_shape[1],
            )

    @staticmethod
    def transform_image(img, M, out_shape):
        if isinstance(img, Sequence):
            assert len(img) == len(M)
            return [
                Transform2D.transform_image(b, m, shape)
                for b, m, shape in zip(img, M, out_shape)
            ]
        else:
            if img.dim() == 2:
                img = img[None, None, ...]
            elif img.dim() == 3:
                img = img[None, ...]
            if M.dim() == 2:
                M = M[None, ...]
            return (warp_affine(img.float(), M, out_shape,
                                mode='nearest').to(img.dtype))
