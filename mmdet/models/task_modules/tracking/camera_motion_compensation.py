# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah


@TASK_UTILS.register_module()
class CameraMotionCompensation:
    """Camera motion compensation.

    Args:
        warp_mode (str): Warp mode in opencv.
            Defaults to 'cv2.MOTION_EUCLIDEAN'.
        num_iters (int): Number of the iterations. Defaults to 50.
        stop_eps (float): Terminate threshold. Defaults to 0.001.
    """

    def __init__(self,
                 warp_mode: str = 'cv2.MOTION_EUCLIDEAN',
                 num_iters: int = 50,
                 stop_eps: float = 0.001):
        self.warp_mode = eval(warp_mode)
        self.num_iters = num_iters
        self.stop_eps = stop_eps

    def get_warp_matrix(self, img: np.ndarray, ref_img: np.ndarray) -> Tensor:
        """Calculate warping matrix between two images."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.num_iters, self.stop_eps)
        cc, warp_matrix = cv2.findTransformECC(img, ref_img, warp_matrix,
                                               self.warp_mode, criteria, None,
                                               1)
        warp_matrix = torch.from_numpy(warp_matrix)
        return warp_matrix

    def warp_bboxes(self, bboxes: Tensor, warp_matrix: Tensor) -> Tensor:
        """Warp bounding boxes according to the warping matrix."""
        tl, br = bboxes[:, :2], bboxes[:, 2:]
        tl = torch.cat((tl, torch.ones(tl.shape[0], 1).to(bboxes.device)),
                       dim=1)
        br = torch.cat((br, torch.ones(tl.shape[0], 1).to(bboxes.device)),
                       dim=1)
        trans_tl = torch.mm(warp_matrix, tl.t()).t()
        trans_br = torch.mm(warp_matrix, br.t()).t()
        trans_bboxes = torch.cat((trans_tl, trans_br), dim=1)
        return trans_bboxes.to(bboxes.device)

    def warp_means(self, means: np.ndarray, warp_matrix: Tensor) -> np.ndarray:
        """Warp track.mean according to the warping matrix."""
        cxcyah = torch.from_numpy(means[:, :4]).float()
        xyxy = bbox_cxcyah_to_xyxy(cxcyah)
        warped_xyxy = self.warp_bboxes(xyxy, warp_matrix)
        warped_cxcyah = bbox_xyxy_to_cxcyah(warped_xyxy).numpy()
        means[:, :4] = warped_cxcyah
        return means

    def track(self, img: Tensor, ref_img: Tensor, tracks: dict,
              num_samples: int, frame_id: int, metainfo: dict) -> dict:
        """Tracking forward."""
        img = img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        ref_img = ref_img.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        warp_matrix = self.get_warp_matrix(img, ref_img)

        # rescale the warp_matrix due to the `resize` in pipeline
        scale_factor_h, scale_factor_w = metainfo['scale_factor']
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale_factor_w
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale_factor_h

        bboxes = []
        num_bboxes = []
        means = []
        for k, v in tracks.items():
            if int(v['frame_ids'][-1]) < frame_id - 1:
                _num = 1
            else:
                _num = min(num_samples, len(v.bboxes))
            num_bboxes.append(_num)
            bboxes.extend(v.bboxes[-_num:])
            if len(v.mean) > 0:
                means.append(v.mean)
        bboxes = torch.cat(bboxes, dim=0)
        warped_bboxes = self.warp_bboxes(bboxes, warp_matrix.to(bboxes.device))

        warped_bboxes = torch.split(warped_bboxes, num_bboxes)
        for b, (k, v) in zip(warped_bboxes, tracks.items()):
            _num = b.shape[0]
            b = torch.split(b, [1] * _num)
            tracks[k].bboxes[-_num:] = b

        if means:
            means = np.asarray(means)
            warped_means = self.warp_means(means, warp_matrix)
            for m, (k, v) in zip(warped_means, tracks.items()):
                tracks[k].mean = m

        return tracks
