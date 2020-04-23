import torch

from .base_bbox_coder import BaseBBoxCoder
from ..registry import BBOX_CODERS


@BBOX_CODERS.register_module
class TBLRBBoxCoder(BaseBBoxCoder):
    """TBLR BBox coder

    Following the practice in FSAF [1]_, this coder encodes bbox (x1, y1, x2,
    y2) into (top, bottom, left, right) and decode it back to the original.

    References:
        .. [1] https://arxiv.org/abs/1903.00621

    Args:
        normalizer (Sequence[float] | float): denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 normalizer=1.0):
        super(BaseBBoxCoder, self).__init__()
        self.normalizer = normalizer

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bboxes2tblr(bboxes, gt_bboxes, self.normalizer)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = tblr2bboxes(bboxes, pred_bboxes, self.normalizer,
                                     max_shape)

        return decoded_bboxes


def bboxes2tblr(priors, gt, normalizer=1.0):
    """Encode ground truth boxes

    Args:
        priors (FloatTensor): Prior boxes in point form
            Shape: [num_proposals,4].
        gt (FloatTensor): Coords of ground truth for each prior in point-form
            Shape: [num_proposals, 4].
        normalizer (list | float): normalization parameter of
            encoded boxes. If it is a FloatTensor, it has to have length = 4

    Return:
        encoded boxes (FloatTensor), Shape: [num_proposals, 4]
    """

    # dist b/t match center and prior's center
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer).to(priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    wh = priors[:, 2:4] - priors[:, 0:2]

    xmin, ymin, xmax, ymax = gt.split(1, dim=1)
    top = prior_centers[:, 1].unsqueeze(1) - ymin
    bottom = ymax - prior_centers[:, 1].unsqueeze(1)
    left = prior_centers[:, 0].unsqueeze(1) - xmin
    right = xmax - prior_centers[:, 0].unsqueeze(1)
    loc = torch.cat((top, bottom, left, right), dim=1)
    w, h = torch.split(wh, 1, dim=1)
    loc[:, :2] /= h
    # convert them to the coordinate on the featuremap: 0 -fm_size
    loc[:, 2:] /= w
    return loc / normalizer


def tblr2bboxes(priors, tblr, normalizer=1.0, max_shape=None):
    """Decode tblr outputs to prediction boxes

    Args:
        priors (FloatTensor): Prior boxes in point form
            Shape: [n,4].
        tblr (FloatTensor): Coords of network output in tblr form
            Shape: [n, 4].
        normalizer (list | float): normalization parameter of encoded boxes
        max_shape (tuple): Shape of the image.

    Return:
        encoded boxes (FloatTensor), Shape: [n, 4]
    """
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer).to(priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'

    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    wh = priors[:, 2:4] - priors[:, 0:2] + 1
    w, h = torch.split(wh, 1, dim=1)
    loc_decode[:, :2] *= h
    loc_decode[:, 2:] *= w
    top, bottom, left, right = loc_decode.split(1, dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if max_shape is not None:
        boxes[:, 0].clamp_(min=0, max=max_shape[1] - 1)
        boxes[:, 1].clamp_(min=0, max=max_shape[0] - 1)
        boxes[:, 2].clamp_(min=0, max=max_shape[1] - 1)
        boxes[:, 3].clamp_(min=0, max=max_shape[0] - 1)
    return boxes
