import mmcv


def split_combined_gt_polys(gt_polys, gt_poly_lens, num_polys_per_mask):
    """Split the combined 1-D polys into masks.

    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.

    Args:
        gt_polys (list): a list (length = image num) of 1-D tensors
        gt_poly_lens (list): a list (length = image num) of poly length
        num_polys_per_mask (list): a list (length = image num) of poly number
            of each mask

    Returns:
        list: a list (length = image num) of list (length = mask num) of
            list (length = poly num) of numpy array
    """
    mask_polys_list = []
    for img_id in range(len(gt_polys)):
        gt_polys_single = gt_polys[img_id].cpu().numpy()
        gt_polys_lens_single = gt_poly_lens[img_id].cpu().numpy().tolist()
        num_polys_per_mask_single = num_polys_per_mask[
            img_id].cpu().numpy().tolist()

        split_gt_polys = mmcv.slice_list(gt_polys_single, gt_polys_lens_single)
        mask_polys = mmcv.slice_list(split_gt_polys, num_polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list
