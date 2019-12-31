import torch


def anchor_offset(anchor_list, anchor_strides, featmap_sizes):
    """ Get offest for deformable conv based on anchor shape
    NOTE: currently support deformable kernel_size=3 and dilation=1

    Args:
        anchor_list (list[list[tensor])): [NI, NLVL, NA, 4] list of
            multi-level anchors
            anchor_strides (list): anchor stride of each level
    Returns:
        offset_list (list[tensor]): [NLVL, NA, 2, 18]: offset of 3x3 deformable
        kernel.
    """

    def _shape_offset(anchors, stride):
        # currently support kernel_size=3 and dilation=1
        ks = 3
        dilation = 1
        pad = (ks - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)  # return order matters
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        w = (anchors[:, 2] - anchors[:, 0] + 1) / stride
        h = (anchors[:, 3] - anchors[:, 1] + 1) / stride
        w = w / (ks - 1) - dilation
        h = h / (ks - 1) - dilation
        offset_x = w[:, None] * xx  # (NA, ks**2)
        offset_y = h[:, None] * yy  # (NA, ks**2)
        return offset_x, offset_y

    def _ctr_offset(anchors, stride, featmap_size):
        feat_h, feat_w = featmap_size
        assert len(anchors) == feat_h * feat_w

        x = (anchors[:, 0] + anchors[:, 2]) * 0.5
        y = (anchors[:, 1] + anchors[:, 3]) * 0.5
        # compute centers on feature map
        x = (x - (stride - 1) * 0.5) / stride
        y = (y - (stride - 1) * 0.5) / stride
        # compute predefine centers
        xx = torch.arange(0, feat_w, device=anchors.device)
        yy = torch.arange(0, feat_h, device=anchors.device)
        yy, xx = torch.meshgrid(yy, xx)
        xx = xx.reshape(-1).type_as(x)
        yy = yy.reshape(-1).type_as(y)

        offset_x = x - xx  # (NA, )
        offset_y = y - yy  # (NA, )
        return offset_x, offset_y

    num_imgs = len(anchor_list)
    num_lvls = len(anchor_list[0])
    dtype = anchor_list[0][0].dtype
    device = anchor_list[0][0].device
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

    offset_list = []
    for i in range(num_imgs):
        mlvl_offset = []
        for lvl in range(num_lvls):
            c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                 anchor_strides[lvl],
                                                 featmap_sizes[lvl])
            s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                   anchor_strides[lvl])

            # offset = ctr_offset + shape_offset
            offset_x = s_offset_x + c_offset_x[:, None]
            offset_y = s_offset_y + c_offset_y[:, None]

            # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
            offset = torch.stack([offset_y, offset_x], dim=-1)
            offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
            mlvl_offset.append(offset)
        offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
    offset_list = images_to_levels(offset_list, num_level_anchors)
    return offset_list


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end])
        start = end
    return level_targets
