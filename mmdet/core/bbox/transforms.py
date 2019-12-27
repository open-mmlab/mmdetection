import mmcv
import numpy as np
import torch


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / new_bboxes.new_tensor(scale_factor)
    return new_bboxes


def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bboxes2tblr(priors, gt, normalizer=1.0):
    """Encode ground truth boxes

    Args:
        priors (FloatTensor): Prior boxes in point form
            Shape: [num_proposals,4].
        gt (FloatTensor): Coords of ground truth for each prior in point-form
            Shape: [num_proposals, 4].
        normalizer (float): normalization parameter of encoded boxes

    Return:
        encoded boxes (FloatTensor), Shape: [num_proposals, 4]
    """

    # dist b/t match center and prior's center
    prior_centers = (priors[:, 0:2] + priors[:, 2:4] + 1) / 2
    wh = priors[:, 2:4] - priors[:, 0:2] + 1

    xmin, ymin, xmax, ymax = gt.split(1, dim=1)
    top = prior_centers[:, 1].unsqueeze(1) - ymin
    bottom = ymax - prior_centers[:, 1].unsqueeze(1) + 1
    left = prior_centers[:, 0].unsqueeze(1) - xmin
    right = xmax - prior_centers[:, 0].unsqueeze(1) + 1
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
        normalizer (float): normalization parameter of encoded boxes
        max_shape (tuple): Shape of the image.

    Return:
        encoded boxes (FloatTensor), Shape: [n, 4]
    """
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4] + 1) / 2
    wh = priors[:, 2:4] - priors[:, 0:2] + 1
    w, h = torch.split(wh, 1, dim=1)
    loc_decode[:, :2] *= h
    loc_decode[:, 2:] *= w
    top, bottom, left, right = loc_decode.split(1, dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right - 1
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom - 1
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if max_shape is not None:
        boxes[:, 0].clamp_(min=0, max=max_shape[1] - 1)
        boxes[:, 1].clamp_(min=0, max=max_shape[0] - 1)
        boxes[:, 2].clamp_(min=0, max=max_shape[1] - 1)
        boxes[:, 3].clamp_(min=0, max=max_shape[0] - 1)
    return boxes
