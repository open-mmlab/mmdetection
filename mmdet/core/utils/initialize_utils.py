import torch.optim as optim


def caffe2_initialize(model, cfg):
    """Initialize lr and weight_decay of the model as caffe2 style.

    The lr of bias is 2 times base lr, the weight decay of bias is 0.

    Args:
        model (obj): network.
        cfg (dict): optimizer cfg.

    Returns:
        optimizer: a optimizer contains param groups based on above setting.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = cfg.copy()
    optimizer_type = optimizer_cfg.pop('type')

    params = []
    base_lr = optimizer_cfg.lr
    base_weight_decay = optimizer_cfg.weight_decay
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        if 'bias' in k:
            lr = base_lr * 2
            weight_decay = 0
        else:
            lr = base_lr
            weight_decay = base_weight_decay
        params.append({"params": [v], "lr": lr, "weight_decay": weight_decay})
    return getattr(optim, optimizer_type)(
        params, base_lr, momentum=optimizer_cfg.momentum)
