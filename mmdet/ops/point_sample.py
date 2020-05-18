import torch
import torch.nn.functional as F


def normalize(grid):
    """
    convert from range [-1, 1] to [0, 1]
    """
    return (grid + 1.0) / 2.0


def denormalize(grid):
    """
    convert from range [0, 1] to [-1, 1]
    """
    return grid * 2.0 - 1.0


def generate_grid(num_grid, size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        num_grid (int): The number of grids to sample, one for each region.
        size (tuple(int, int)): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (num_grid, size[0]*size[1], 2) that
        contains coordinates for the regular grids.
    """
    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device)
    grid = F.affine_grid(
        affine_trans, torch.Size((1, 1, *size)), align_corners=False)
    grid = normalize(grid)
    return grid.view(1, -1, 2).expand(num_grid, -1, -1)


def rel_roi_point2abs_img_point(rois, rel_roi_points):
    """ Convert roi based relative point coordinates to image based absolute
        point coordinates.

    Args:
        rois (Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (Tensor): Point coordinates inside RoI, relative to
            RoI, location, range (0, 1), shape (N, P, 2)

    Returns:
        Tensor: Image based absolute point coordinates, shape (N, P, 2)
    """
    with torch.no_grad():
        assert rel_roi_points.size(0) == rois.size(0)
        assert rois.dim() == 2
        assert rel_roi_points.dim() == 3
        assert rel_roi_points.size(2) == 2
        # remove batch idx
        if rois.size(1) == 5:
            rois = rois[:, 1:]
        abs_img_points = rel_roi_points.clone()
        abs_img_points[:, :, 0] = abs_img_points[:, :, 0] * (
            rois[:, None, 2] - rois[:, None, 0])
        abs_img_points[:, :, 1] = abs_img_points[:, :, 1] * (
            rois[:, None, 3] - rois[:, None, 1])
        abs_img_points[:, :, 0] += rois[:, None, 0]
        abs_img_points[:, :, 1] += rois[:, None, 1]
    return abs_img_points


def abs_img_point2rel_img_point(abs_img_points, img_shape, spatial_scale=1):
    """ Convert image based absolute point coordinates to image based relative
        coordinates for sampling.

    Args:
        abs_img_points (Tensor): Image based absolute point coordinates,
            shape (N, P, 2)
        img_shape (tuple): (height, width) of image or feature map.
        spatial_scale (int): Scale factor of input w.r.t. original image.
            Default: 1

    Returns:
        Tensor: Image based relative point coordinates for sampling,
            shape (N, P, 2)
    """
    assert isinstance(img_shape, tuple) and len(img_shape) == 2
    h, w = img_shape
    scale = torch.tensor([w, h],
                         dtype=torch.float,
                         device=abs_img_points.device) * spatial_scale
    scale = scale.view(1, 1, 2)
    rel_img_points = abs_img_points / scale

    return rel_img_points


def rel_roi_point2rel_img_point(rois,
                                rel_roi_points,
                                img_shape,
                                spatial_scale=1):
    """ Convert roi based relative point coordinates to image based absolute
        point coordinates.

    Args:
        rois (Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (Tensor): Point coordinates inside RoI, relative to
            RoI, location, range (0, 1), shape (N, P, 2)
        img_shape (tuple): (height, width) of image or feature map.
        spatial_scale (int): Scale factor of input w.r.t. original image.
            Default: 1

    Returns:
        Tensor: Image based relative point coordinates for sampling,
            shape (N, P, 2)
    """
    abs_img_point = rel_roi_point2abs_img_point(rois, rel_roi_points)
    rel_img_point = abs_img_point2rel_img_point(abs_img_point, img_shape,
                                                spatial_scale)

    return rel_img_point


def point_sample(input, points, align_corners=False, **kwargs):
    """
    A wrapper around :function:`grid_sample` to support 3D point_coords tensors
    Unlike :function:`torch.nn.functional.grid_sample` it assumes point_coords
    to lie inside [0, 1] x [0, 1] square.

    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False

    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """
    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    output = F.grid_sample(
        input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output
