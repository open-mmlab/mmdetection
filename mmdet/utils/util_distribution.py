# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

USE_DEVICE = ''

dp_factory = {
        'cuda' : MMDataParallel
    }

ddp_factory = {
        'cuda' : MMDistributedDataParallel
    }

def build_dp(model, device = 'cuda'):
    assert device in ['cuda', 'mlu'], "Only available for cuda or mlu devices."
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    return model, dp_factory[device]

def build_ddp(model, device = 'cuda'):
    assert device in ['cuda', 'mlu'], "Only available for cuda or mlu devices."
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        from mmcv.device.mlu import  MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return model, ddp_factory[device]

def select_device():
    global USE_DEVICE
    if USE_DEVICE != '' :
        return USE_DEVICE
    is_device_available = {'cuda': torch.cuda.is_available(),
                       'mlu': hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()}
    # st_device = dict(filter(lambda x : x[1] , is_device_available.items()))
    st_device = [k for k, v in is_device_available.items() if v ]
    assert len(st_device) == 1, "Only one device type is available, please check!"
    USE_DEVICE = st_device[0]
    return USE_DEVICE
