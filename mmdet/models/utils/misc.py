from torch.nn import functional as F


def upsample_like(logits, target, mode='bilinear', align_corners=False):
    """Up-sample the logits to the shape of target.

    The input logits can be 3D or 4D, the shape of the target needs to be in
    this form: (..., h, w).
    """
    assert len(target.shape) >= 2

    def _upsample_like(source, target, mode='bilinear', align_corners=False):
        """Upsample the logits (4D) to the shape of target."""
        h, w = target.shape[-2:]
        lh, lw = source.shape[-2:]
        if h != lh or w != lw:
            source = F.interpolate(
                source, size=(h, w), mode=mode, align_corners=align_corners)
        return source

    if len(logits.shape) == 3:
        logits = logits[:, None, :, :]
        logits = _upsample_like(logits, target, mode, align_corners)
        return logits[:, 0, :, :]
    else:
        return _upsample_like(logits, target, mode, align_corners)
