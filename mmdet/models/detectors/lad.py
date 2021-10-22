import torch
from mmcv.runner import _load_checkpoint

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class LAD(SingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_backbone,
                 teacher_neck,
                 teacher_bbox_head,
                 teacher_ckpt,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(LAD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
        self.teacher_backbone = build_backbone(teacher_backbone)
        if teacher_neck is not None:
            self.teacher_neck = build_neck(teacher_neck)
        teacher_bbox_head.update(train_cfg=train_cfg)
        teacher_bbox_head.update(test_cfg=test_cfg)
        self.teacher_bbox_head = build_head(teacher_bbox_head)
        self.init_teacher_weights(teacher_ckpt)

    @property
    def with_teacher_neck(self):
        """bool: whether the detector has a teacher_neck"""
        return hasattr(self, 'teacher_neck') and self.teacher_neck is not None

    def init_teacher_weights(self, ckpt_file):
        """Load checkpoint for teacher."""
        ckpt = _load_checkpoint(ckpt_file, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        teacher_ckpt = dict()
        for key in ckpt:
            teacher_ckpt['teacher_' + key] = ckpt[key]
        self.load_state_dict(teacher_ckpt, strict=False)

    def teacher_eval(self):
        """Eval teacher."""
        self.teacher_backbone.eval()
        if self.with_teacher_neck:
            self.teacher_neck.eval()
        self.teacher_bbox_head.eval()

    def extract_teacher_feat(self, img):
        """Directly extract teacher features from the backbone+neck."""
        x = self.teacher_backbone(img)
        if self.with_teacher_neck:
            x = self.teacher_neck(x)
        return x

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
        super(SingleStageDetector, self).forward_train(img, img_metas)

        # teacher infers Label Assignment results
        with torch.no_grad():
            # MUST force teacher to `eval` every training step, b/c at the
            # beginning of epoch, the runner calls all nn.Module elements
            # to be `train`
            self.teacher_eval()

            # assignment result is obtained based on only teacher
            x_teacher = self.extract_teacher_feat(img)
            outs_teacher = self.teacher_bbox_head(x_teacher)
            la_results = self.teacher_bbox_head.get_la(*outs_teacher,
                                                       gt_bboxes, gt_labels,
                                                       img_metas,
                                                       gt_bboxes_ignore)

        # student receives the assignment results to learn
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, la_results, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)

        return losses
