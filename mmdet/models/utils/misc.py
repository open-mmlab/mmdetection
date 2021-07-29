from torch.nn import functional as F


def upsample_like(logits, target, mode='bilinear', align_corners=False):
    """Up-sample the logits to the shape of target.

    Up-sample the logits to the shape of target. The input must be a Tensor,
    but the target can be a Tensor or a np.ndarray with the shape
    (..., target_h, target_w).

    Args:
        logits (Tensor): The 3D/4D input logits with the shape (N, H, W) or
            (N, C, H, W).
        target (Tensor | np.ndarray): The upsampling target with the shape
            (..., target_h, target_w).
        mode (str): Algorithm used for upsampling. The options are the same
            as those in F.interpolate(). Default: ``'bilinear'``.
        align_corners (bool): The same as the argument in F.interpolate().

    Returns:
        Tensor: The up-sampled logits.
    """
    assert len(target.shape) >= 2

    def _upsample_like(origin, target, mode='bilinear', align_corners=False):
        """Upsample the logits (4D) to the shape of target."""
        target_h, target_w = target.shape[-2:]
        origin_h, origin_w = origin.shape[-2:]
        if target_h != origin_h or target_w != origin_w:
            origin = F.interpolate(
                origin,
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners)
        return origin

    if len(logits.shape) == 3:
        logits = logits[:, None, :, :]
        logits = _upsample_like(logits, target, mode, align_corners)
        return logits[:, 0, :, :]
    else:
        return _upsample_like(logits, target, mode, align_corners)
