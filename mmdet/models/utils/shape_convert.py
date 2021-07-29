def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before convertion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after convertion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before convertion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after convertion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()
