import numpy as np
import torch

from .gpu_nms import gpu_nms
from .cpu_nms import cpu_nms
from .cpu_soft_nms import cpu_soft_nms


def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        if dets.is_cuda:
            device_id = dets.get_device()
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    if dets_np.shape[0] == 0:
        inds = []
    else:
        inds = (gpu_nms(dets_np, iou_thr, device_id=device_id)
                if device_id is not None else cpu_nms(dets_np, iou_thr))

    if is_tensor:
        inds = dets.new_tensor(inds, dtype=torch.long)
    else:
        inds = np.array(inds, dtype=np.int64)
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = cpu_soft_nms(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
