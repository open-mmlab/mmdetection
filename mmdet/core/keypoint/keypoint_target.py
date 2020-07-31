import numpy as np
import torch


def compute_nme(preds, target, normalize=False):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    is_visible = target[:, :, 2] == 2
    preds = preds[:, :, :2]
    target = target[:, :, :2]

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt, v = preds[i, ], target[i, ], is_visible[i, ]
        _rmse = [
            np.linalg.norm(pred - gt)
            for (pred, gt, _v) in zip(pts_pred, pts_gt, v) if (_v)
        ]
        rmse[i] = np.sum(_rmse) / max(1, len(_rmse))
        if normalize:
            assert L == 98, 'Unknown interocular distance'
            if L == 98:  # WFLW dataset
                interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
            rmse[i] /= interocular
    return rmse
