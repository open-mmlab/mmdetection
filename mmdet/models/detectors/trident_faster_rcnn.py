from ..builder import DETECTORS
from .faster_rcnn import FasterRCNN


@DETECTORS.register_module()
class TridentFasterRCNN(FasterRCNN):

    def __init__(self,
                 num_branch,
                 test_branch_idx,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):

        backbone['num_branch'] = num_branch
        backbone['test_branch_idx'] = test_branch_idx
        roi_head['num_branch'] = num_branch
        roi_head['test_branch_idx'] = test_branch_idx
        self.num_branch = num_branch
        self.test_branch_idx = test_branch_idx
        super(TridentFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def simple_test_rpn(self, x, img_metas):
        """make copies of img and gts to fit multi-branch."""
        num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        trident_img_metas = img_metas * num_branch
        return super(TridentFasterRCNN,
                     self).simple_test_rpn(x, trident_img_metas)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        """make copies of img and gts to fit multi-branch."""
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)

        return super(TridentFasterRCNN,
                     self).forward_train(img, trident_img_metas,
                                         trident_gt_bboxes, trident_gt_labels)
