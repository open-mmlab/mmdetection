# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pycocotools.mask as mask_util
import torch
from mmengine.utils import slice_list


def split_combined_polys(polys, poly_lens, polys_per_mask):
    """Split the combined 1-D polys into masks.

    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.

    Args:
        polys (list): a list (length = image num) of 1-D tensors
        poly_lens (list): a list (length = image num) of poly length
        polys_per_mask (list): a list (length = image num) of poly number
            of each mask

    Returns:
        list: a list (length = image num) of list (length = mask num) of \
            list (length = poly num) of numpy array.
    """
    mask_polys_list = []
    for img_id in range(len(polys)):
        polys_single = polys[img_id]
        polys_lens_single = poly_lens[img_id].tolist()
        polys_per_mask_single = polys_per_mask[img_id].tolist()

        split_polys = slice_list(polys_single, polys_lens_single)
        mask_polys = slice_list(split_polys, polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list


# TODO: move this function to more proper place
def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F',
                         dtype='uint8'))[0])  # encoded with RLE
    return encoded_mask_results


def mask2bbox(masks):
    """Obtain tight bounding boxes of binary masks.

    Args:
        masks (Tensor): Binary mask of shape (n, h, w).

    Returns:
        Tensor: Bboxe with shape (n, 4) of \
            positive region in binary mask.
    """
    N = masks.shape[0]
    bboxes = masks.new_zeros((N, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(N):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1])

    return bboxes
