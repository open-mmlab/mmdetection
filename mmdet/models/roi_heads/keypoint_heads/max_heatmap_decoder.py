import math
from collections.abc import Sequence

import torch

from mmdet.models.builder import HEADS


@HEADS.register_module()
class HeatmapDecodeOneKeypoint():
    """Decodes a heatmap to return a keypoint Only consider the highest
    intensity value, does not handle a 2 keypoints case."""

    def __init__(self, upscale=4, score_th=-1):
        if not isinstance(upscale, Sequence):
            upscale = [upscale]
        if len(upscale) == 1:
            upscale = [upscale[0], upscale[0]]
        self.upscale = torch.tensor(upscale)
        self.score_th = score_th

    def init_weights(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self._decode_heatmap(x)

    def _decode_heatmap(self, output):
        coords = self._get_preds(output)  # float type
        coords = coords.cpu()
        confs = torch.zeros_like(coords[:, :, 0])
        res = output.size()[2:]
        # post-processing
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = output[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if (px >= 0) and (px < res[1]) and (py >= 0) and (py < res[0]):
                    px_m, px_p = max(0, px - 1), min(res[0] - 1, px + 1)
                    py_m, py_p = max(0, py - 1), min(res[1] - 1, py + 1)
                    diff = torch.Tensor([
                        hm[py][px_p] - hm[py][px_m],
                        hm[py_p][px] - hm[py_m][px]
                    ])
                    coords[n][p] += (diff * self.upscale).abs().ceil(
                    ) * diff.sign() / self.upscale
                    confs[n][p] = hm[py, px]
                    for c in range(2):
                        coords[n, p, c] = torch.clamp(coords[n, p, c], 0,
                                                      res[c])
        preds = coords.clone()

        # Transform back
        for i in range(coords.size(0)):
            preds[i] = self._transform_preds(coords[i])

        if preds.dim() < 3:
            preds = preds.view(1, preds.size())

        low_conf = confs < self.score_th
        confs[low_conf] = 0.
        confs_shape = (n + 1, p + 1, 1)
        low_conf = low_conf.reshape(confs_shape).repeat(1, 1, 2)
        preds[low_conf] = -1
        preds = torch.cat((preds, confs.reshape(confs_shape)), axis=2)
        return preds

    def _get_preds(self, scores, min_conf=0):
        """get predictions from score maps in torch Tensor."""
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(
            scores.view(scores.size(0), scores.size(1), -1), 2)
        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1
        preds = idx.repeat(1, 1, 2).float()
        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3)
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3))
        pred_mask = maxval.gt(min_conf).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def _transform_preds(self, coords):
        coords[:, 0:2] = coords[:, 0:2] * self.upscale
        return coords
