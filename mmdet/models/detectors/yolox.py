# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.runner import get_dist_info
import torch.nn.functional as F
import torch
import random
import torch.distributed as dist


@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 input_size=(640, 640),
                 random_size_range=(15, 25),
                 random_size_interval=10):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self.input_size = input_size
        self.random_size_range = random_size_range
        self.random_size_interval = random_size_interval
        self.progress_in_iter = 0

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes = self.preprocess(img, gt_bboxes)

        losses = super(YOLOX, self).forward_train(img, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)

        # random resizing
        if (self.progress_in_iter + 1) % self.random_size_interval == 0:
            self.input_size = self.random_resize()
        self.progress_in_iter += 1

        return losses

    def preprocess(self, img, gt_bboxes):
        scale_y = self.input_size[0] / self._default_input_size[0]
        scale_x = self.input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img, size=self.input_size, mode="bilinear", align_corners=False
            )
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def random_resize(self):
        tensor = torch.LongTensor(2).cuda()

        if self.rank == 0:
            size = random.randint(*self.random_size_range)
            size = (32 * size, 32 * size)
            tensor[0] = size[0]
            tensor[1] = size[1]

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
