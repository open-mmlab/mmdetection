from torch.nn import functional as F


def upsample_like(logits, target):
    assert len(target.shape) >= 2

    def _upsample_like(source, target):
        """Upsample the logits (4D) to the shape of target."""
        h, w = target.shape[-2:]
        lh, lw = source.shape[-2:]
        if h != lh or w != lw:
            source = F.interpolate(
                source, size=(h, w), mode='bilinear', align_corners=False)
        return source

    if len(logits.shape) == 3:
        logits = logits[:, None, :, :]
        logits = _upsample_like(logits, target)
        return logits[:, 0, :, :]
    else:
        return _upsample_like(logits, target)
