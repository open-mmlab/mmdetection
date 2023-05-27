# Copyright (c) OpenMMLab. All rights reserved.
from math import sqrt

import torch
import torch.nn.functional as F


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """生成[2r+1, 2r+1]尺寸的二维高斯核.中间位置为1,向四周递减逼近至0

    Args:
        radius (int): 高斯核半径.
        sigma (int): 高斯函数的 Sigma参数. 默认: 1.
        dtype (torch.dtype): 高斯张量的类型. 默认: torch.float32.
        device (str): 高斯张量生成的设备. 默认: 'cpu'.

    Returns:
        h (Tensor): 高斯核,[2*radius+1, 2*radius+1]
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    # 这个h会生成一个三维的高斯分布图,需要关注x=y=0时,h=1,x=y=r时,h接近于0,
    # 以及表达式中有负号,即指数函数始终在非正半轴.这样就会方便明白h的分布规律了.
    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
    # 将特别小的值设为0,参考 https://zhuanlan.zhihu.com/p/76870605
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """生成heatmap上指定位置(center)生成二维高斯分布的热力图.如在生成位置附近已存在分布则取最大值.
        值得注意的是,该高斯分布的热力图在x,y方向是一致的.
        如果一个gt box高宽不一致,则热力图可能不太能匹配该gt

    Args:
        heatmap (Tensor): 输入 heatmap,[h, w],全为零, 高斯核将覆盖它并保持最大值.
        center (list[int]): 高斯核中心坐标.gt中心坐标的左上角索引(特征图尺寸下)
        radius (int): 高斯核半径.
        k (int): 高斯核系数. 默认: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    # 让直径为奇数的原因是为了后续生成高斯核时能有中心位置.
    diameter = 2 * radius + 1
    # [2*radius+1, 2*radius+1], 这里的这个sigma也是一个超参数
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    # 获取center坐标向上下左右方向辐射的最大长度(不超出特征图尺寸)
    # 因为索引左闭右开,所以right与bottom的计算过程中radius需要+1,是为了保证能取到radius
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    # 需要注意的是,下面两行等号右边的shape都是一致的,因为其切片范围是一致的.
    # [y,x]是gt中心所属grid左上角坐标,  [radius,radius]是gaussian_kernel热力图中心坐标
    # 因为要将生成的二维高斯热力图复制到heatmap上去,[y, x]与[radius,radius]对齐
    # 为了保证复制后能超出heatmap范围.top/bottom/left/right都已做了限制.
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    # 原地修改,此处是当由两个gt box的高斯核部分存在重合时,保存较大值.
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def gaussian_radius(det_size, min_overlap):
    r"""生成二维高斯半径.

    这个方法是修改于
    <https://github.com/princeton-vl/CornerNet-Lite/blob/master/core/sample/
    utils.py#L65>`_.

    给定``min_overlap``之后, 半径可以根据 Vieta 公式通过二次方程计算.

    计算高斯半径有3种情况,具体如下:

    - 图说明：``lt``和``br``表示ground truth box的左上角和右下角.
      ``x`` 表示当``radius=r`` 时在有限位置生成的角.

    - 情况1: 一个角在gt框里面,另一个在外面.

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

    确保生成框和gt框的IoU大于``min_overlap``:

    .. math::
        \cfrac{(w-r)*(h-r)}{w*h+(w+h)r-r^2} \ge {iou} \quad\Rightarrow\quad
        {r^2-(w+h)r+\cfrac{1-iou}{1+iou}*w*h} \ge 0 \\
        {a} = 1,\quad{b} = {-(w+h)},\quad{c} = {\cfrac{1-iou}{1+iou}*w*h} \\
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - 情况2: 两个角都在 gt 框内.

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

    确保生成框和gt框的IoU大于``min_overlap``:

    .. math::
        \cfrac{(w-2*r)*(h-2*r)}{w*h} \ge {iou} \quad\Rightarrow\quad
        {4r^2-2(w+h)r+(1-iou)*w*h} \ge 0 \\
        {a} = 4,\quad {b} = {-2(w+h)},\quad {c} = {(1-iou)*w*h} \\
        {r} \le \cfrac{-b-\sqrt{b^2-4*a*c}}{2*a}

    - 情况3: 两个角都在gt框外.

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

    确保生成框和gt框的IoU大于``min_overlap``:

    .. math::
        \cfrac{w*h}{(w+2*r)*(h+2*r)} \ge {iou} \quad\Rightarrow\quad
        {4*iou*r^2+2*iou*(w+h)r+(iou-1)*w*h} \le 0 \\
        {a} = {4*iou},\quad {b} = {2*iou*(w+h)},\quad {c} = {(iou-1)*w*h} \\
        {r} \le \cfrac{-b+\sqrt{b^2-4*a*c}}{2*a}

    Args:
        det_size (list[int]): 物体的尺寸,[h, w].
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
    # 令r1,r2,r3分别对应于情况1,情况2,情况3时的半径.同时r3为最小半径
    # 如果取得非最小r(比如r2)作为高斯半径,那么在情况3下生成的热力图会使得实际IOU小于
    # min_overlap.所以这里要返回最小的r,以使3种情况的IOU都能大于等于min_overlap.
    return min(r1, r2, r3)


def get_local_maximum(heat, kernel=3):
    """使用指定核大小的 max_pool2d 提取局部最大像素.

    Args:
        heat (Tensor): 需要进行处理的heatmap.
        kernel (int): max pooling的核大小. 默认: 3.

    Returns:
        heat (Tensor): 局部最大像素保持最大值,其他位置为0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """从heatmap中截取前 k 个位置.

    Args:
        scores (Tensor): 需要进行处理的heatmap. [1, num_classes, h, w].
        k (int): 截取指定数量个位置. 默认: 20.

    Returns:
        tuple[torch.Tensor]: top_k个关键点的score, h*w维度的索引, 类别索引,y/x坐标.

        - topk_scores (Tensor): top_k个关键点的score.
        - topk_inds (Tensor): top_k个关键点的h*w维度的索引.
        - topk_clses (Tensor): top_k个关键点的h*w维度的类别索引.
        - topk_ys (Tensor): top_k个关键点的h*w维度的y坐标.
        - topk_xs (Tensor): top_k个关键点的h*w维度的x坐标.
    """
    batch, _, height, width = scores.size()
    # scores -> [bs, nc, h, w] -> [bs, nc * h * w]
    # [bs, k], 不同的行代表不同的图片,每行代表当前图片中前k个最大值,索引
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    # [bs, k](下同)个位置所属的类别
    topk_clses = torch.div(topk_inds, height * width, rounding_mode='floor')
    topk_inds = topk_inds % (height * width)  # 在h*w维度的索引
    topk_ys = torch.div(topk_inds, width, rounding_mode='floor')  # 所属的y坐标
    topk_xs = (topk_inds % width).int().float()  # 所属的x坐标
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    """根据ind获取feat的值.

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
    """转换并获取指定索引上的特征值.

    Args:
        feat (Tensor): Target feature map. [bs,c,h,w]
        ind (Tensor): Target coord index. [bs,k], k∈[0,h*w-1]

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    # [bs, c, h, w] -> [bs, h, w, c] ->  -> [bs, h*w, c]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat
