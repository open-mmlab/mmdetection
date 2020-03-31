import numpy as np
import torch


def ensure_rng(rng=None):
    """
    Simple version of the ``kwarray.ensure_rng``

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng


def random_boxes(num=1, scale=1, rng=None):
    """
    Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes
