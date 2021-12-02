# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt

import torch
import torch.nn.functional as F


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def gaussian_radius(det_size, min_overlap):
    r"""Generate 2D gaussian radius.

    This function is modified from the `official github repo
    <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
    utils.py#L65>`_.

    Given ``min_overlap``, radius could computed by a quadratic equation
    according to Vieta's formulas.

    There are 3 cases for computing gaussian radius, details are following:

    - Explanation of figure: ``lt`` and ``br`` indicates the left-top and
      bottom-right corner of ground truth box. ``x`` indicates the
      generated corner at the limited position when ``radius=r``.

    - Case1: one corner is inside the gt box and the other is outside.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x----------+--+
        |  |          |  |
        |  |          |  |    height
        |  | overlap  |  |
        |  |          |  |
        |  |          |  |      v
        +--+---------br--+      -
           |          |  |
           +----------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
        {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
        {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case2: both two corners are inside the gt box.

    .. code:: text

        |<   width   >|

        lt-+----------+         -
        |  |          |         ^
        +--x-------+  |
        |  |       |  |
        |  |overlap|  |       height
        |  |       |  |
        |  +-------x--+
        |          |  |         v
        +----------+-br         -

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
        {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
        {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h}
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - Case3: both two corners are outside the gt box.

    .. code:: text

           |<   width   >|

        x--+----------------+
        |  |                |
        +-lt-------------+  |   -
        |  |             |  |   ^
        |  |             |  |
        |  |   overlap   |  | height
        |  |             |  |
        |  |             |  |   v
        |  +------------br--+   -
        |                |  |
        +----------------+--x

    To ensure IoU of generated box and gt box is larger than ``min_overlap``:

    .. math::
        \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
        {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
        {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
        {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

    Args:
        det_size (list[int]): Shape of object.
        min_overlap (float): Min IoU with ground truth for boxes generated by
            keypoints inside the gaussian kernel.

    Returns:
        radius (int): Radius of gaussian kernel.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat
