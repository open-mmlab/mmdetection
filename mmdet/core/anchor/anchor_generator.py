import torch

from .registry import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module
class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors

    Attributes:
        base_size: The basic size of anchor.
        scales: Anchor scales.
        ratios (list[float]): The list of ratios between the height and width
            of anchors.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales.
        center (float | None): The center of the anchor relative to the feature
            grid center.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])
    """

    def __init__(self,
                 base_size,
                 scales,
                 ratios,
                 scale_major=True,
                 center=None,
                 center_offset=0.):
        if center_offset != 0:
            assert center is None, 'center cannot be set when center_offset=' \
                '0, {} is given.'.format(center)
        assert 0 <= center_offset <= 1, 'center_offset should be in range ' \
            '[0, 1], {} is given.'.format(center_offset)

        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.center = center
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = self.center

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:,
                      None].expand(valid.size(0),
                                   self.num_base_anchors).contiguous().view(-1)
        return valid


@ANCHOR_GENERATORS.register_module
class LegacyAnchorGenerator(AnchorGenerator):
    """Legacy anchor generator used in MMDetection V1.x

    Difference to the V2.0 anchor generator:
    1. The center offset of V1.x anchors are set to be 0.5 rather than 0.
    2. The width/height are minused by 1 when calculating the anchors' centers
       and corners to meet the V1.x coordinate system.
    3. The anchors' corners are quantized.

    Attributes:
        base_size: The basic size of anchor.
        scales: Anchor scales.
        ratios (list[float]): The list of ratios between the height and width
            of anchors.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales.
        center (float | None): The center of the anchor relative to the feature
            grid center.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. Use 0.5 for V1.0.

    Examples:
        >>> from mmdet.core import LegacyAnchorGenerator
        >>> self = LegacyAnchorGenerator(9, [1.], [1.], center_offset=0.5)
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.center is None:
            x_center = self.center_offset * (w - 1)
            y_center = self.center_offset * (h - 1)
        else:
            x_center, y_center = self.center

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * (ws - 1), y_center - 0.5 * (hs - 1),
            x_center + 0.5 * (ws - 1), y_center + 0.5 * (hs - 1)
        ]
        base_anchors = torch.stack(base_anchors, dim=-1).round()

        return base_anchors
