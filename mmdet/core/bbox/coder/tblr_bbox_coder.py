import mmcv
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class TBLRBBoxCoder(BaseBBoxCoder):
    """TBLR BBox coder.

    Following the practice in `FSAF <https://arxiv.org/abs/1903.00621>`_,
    this coder encodes gt bboxes (x1, y1, x2, y2) into (top, bottom, left,
    right) and decode it back to the original.

    Args:
        normalizer (list | float): Normalization factor to be
          divided with when coding the coordinates. If it is a list, it should
          have length of 4 indicating normalization factor in tblr dims.
          Otherwise it is a unified float factor for all dims. Default: 4.0
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self, normalizer=4.0, clip_border=True):
        super(BaseBBoxCoder, self).__init__()
        self.normalizer = normalizer
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes`` in the (top, left,
        bottom, right) order.

        Args:
            bboxes (torch.Tensor): source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): target of the transformation, e.g.,
                ground truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bboxes2tblr(
            bboxes, gt_bboxes, normalizer=self.normalizer)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes.Shape (B, N, 4) or (N, 4)
            pred_bboxes (torch.Tensor): Encoded boxes with shape
               (B, N, 4) or (N, 4)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        decoded_bboxes = tblr2bboxes(
            bboxes,
            pred_bboxes,
            normalizer=self.normalizer,
            max_shape=max_shape,
            clip_border=self.clip_border)

        return decoded_bboxes


@mmcv.jit(coderize=True)
def bboxes2tblr(priors, gts, normalizer=4.0, normalize_by_wh=True):
    """Encode ground truth boxes to tblr coordinate.

    It first convert the gt coordinate to tblr format,
     (top, bottom, left, right), relative to prior box centers.
     The tblr coordinate may be normalized by the side length of prior bboxes
     if `normalize_by_wh` is specified as True, and it is then normalized by
     the `normalizer` factor.

    Args:
        priors (Tensor): Prior boxes in point form
            Shape: (num_proposals,4).
        gts (Tensor): Coords of ground truth for each prior in point-form
            Shape: (num_proposals, 4).
        normalizer (Sequence[float] | float): normalization parameter of
            encoded boxes. If it is a list, it has to have length = 4.
            Default: 4.0
        normalize_by_wh (bool): Whether to normalize tblr coordinate by the
            side length (wh) of prior bboxes.

    Return:
        encoded boxes (Tensor), Shape: (num_proposals, 4)
    """

    # dist b/t match center and prior's center
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == gts.size(0)
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    xmin, ymin, xmax, ymax = gts.split(1, dim=1)
    top = prior_centers[:, 1].unsqueeze(1) - ymin
    bottom = ymax - prior_centers[:, 1].unsqueeze(1)
    left = prior_centers[:, 0].unsqueeze(1) - xmin
    right = xmax - prior_centers[:, 0].unsqueeze(1)
    loc = torch.cat((top, bottom, left, right), dim=1)
    if normalize_by_wh:
        # Normalize tblr by anchor width and height
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc[:, :2] /= h  # tb is normalized by h
        loc[:, 2:] /= w  # lr is normalized by w
    # Normalize tblr by the given normalization factor
    return loc / normalizer


@mmcv.jit(coderize=True)
def tblr2bboxes(priors,
                tblr,
                normalizer=4.0,
                normalize_by_wh=True,
                max_shape=None,
                clip_border=True):
    """Decode tblr outputs to prediction boxes.

    The process includes 3 steps: 1) De-normalize tblr coordinates by
    multiplying it with `normalizer`; 2) De-normalize tblr coordinates by the
    prior bbox width and height if `normalize_by_wh` is `True`; 3) Convert
    tblr (top, bottom, left, right) pair relative to the center of priors back
    to (xmin, ymin, xmax, ymax) coordinate.

    Args:
        priors (Tensor): Prior boxes in point form (x0, y0, x1, y1)
          Shape: (N,4) or (B, N, 4).
        tblr (Tensor): Coords of network output in tblr form
          Shape: (N, 4) or (B, N, 4).
        normalizer (Sequence[float] | float): Normalization parameter of
          encoded boxes. By list, it represents the normalization factors at
          tblr dims. By float, it is the unified normalization factor at all
          dims. Default: 4.0
        normalize_by_wh (bool): Whether the tblr coordinates have been
          normalized by the side length (wh) of prior bboxes.
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If priors shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Return:
        encoded boxes (Tensor): Boxes with shape (N, 4) or (B, N, 4)
    """
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)
    if priors.ndim == 3:
        assert priors.size(1) == tblr.size(1)

    loc_decode = tblr * normalizer
    prior_centers = (priors[..., 0:2] + priors[..., 2:4]) / 2
    if normalize_by_wh:
        wh = priors[..., 2:4] - priors[..., 0:2]
        w, h = torch.split(wh, 1, dim=-1)
        # Inplace operation with slice would failed for exporting to ONNX
        th = h * loc_decode[..., :2]  # tb
        tw = w * loc_decode[..., 2:]  # lr
        loc_decode = torch.cat([th, tw], dim=-1)
    # Cannot be exported using onnx when loc_decode.split(1, dim=-1)
    top, bottom, left, right = loc_decode.split((1, 1, 1, 1), dim=-1)
    xmin = prior_centers[..., 0].unsqueeze(-1) - left
    xmax = prior_centers[..., 0].unsqueeze(-1) + right
    ymin = prior_centers[..., 1].unsqueeze(-1) - top
    ymax = prior_centers[..., 1].unsqueeze(-1) + bottom

    bboxes = torch.cat((xmin, ymin, xmax, ymax), dim=-1)

    if clip_border and max_shape is not None:
        if not isinstance(max_shape, torch.Tensor):
            max_shape = priors.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(priors)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = priors.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape],
                           dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes
