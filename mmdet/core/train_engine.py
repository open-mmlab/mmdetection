import numpy as np
import torch
from collections import OrderedDict
from mmdet.nn.parallel import scatter


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_key, loss_value in losses.items():
        if isinstance(loss_value, dict):
            for _key, _value in loss_value.items():
                if isinstance(_value, list):
                    _value = sum([_loss.mean() for _loss in _value])
                else:
                    _value = _value.mean()
                log_vars[_keys] = _value
        elif isinstance(loss_value, list):
            log_vars[loss_key] = sum(_loss.mean() for _loss in loss_value)
        else:
            log_vars[loss_key] = loss_value.mean()

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    for _key, _value in log_vars.items():
        log_vars[_key] = _value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode, args=None):
    data = scatter(data, [torch.cuda.current_device()])[0]
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss / args.world_size,
        log_vars=log_vars,
        num_samples=len(data['img'].data))

    return outputs
