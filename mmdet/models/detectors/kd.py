from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class KnowledgeDistillationSingleStageDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config='',
                 teacher_model='',
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)

        from mmdet.apis.inference import init_detector

        self.teacher_model = init_detector(
            teacher_config, teacher_model, device=torch.cuda.current_device())

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
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x)

        losses = self.bbox_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            out_teacher,
            gt_labels,
            gt_bboxes_ignore,
        )
        return losses

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        # didn't work, still unused parameters error
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        SingleStageDetector.__setattr__(self, name, value)
