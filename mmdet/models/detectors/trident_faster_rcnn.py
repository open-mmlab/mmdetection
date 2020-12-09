from ..builder import DETECTORS
from .faster_rcnn import FasterRCNN


@DETECTORS.register_module()
class TridentFasterRCNN(FasterRCNN):
    """Implementation of `TridentNet <https://arxiv.org/abs/1901.01892>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):

        super(TridentFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        assert self.backbone.num_branch == self.roi_head.num_branch
        assert self.backbone.test_branch_idx == self.roi_head.test_branch_idx
        self.num_branch = self.backbone.num_branch
        self.test_branch_idx = self.backbone.test_branch_idx

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
            trident_img_metas = img_metas * num_branch
            proposal_list = self.rpn_head.simple_test_rpn(x, trident_img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, trident_img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        num_branch = (self.num_branch if self.test_branch_idx == -1 else 1)
        trident_img_metas = [img_metas * num_branch for img_metas in img_metas]
        proposal_list = self.rpn_head.aug_test_rpn(x, trident_img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, **kwargs):
        """make copies of img and gts to fit multi-branch."""
        trident_gt_bboxes = tuple(gt_bboxes * self.num_branch)
        trident_gt_labels = tuple(gt_labels * self.num_branch)
        trident_img_metas = tuple(img_metas * self.num_branch)

        return super(TridentFasterRCNN,
                     self).forward_train(img, trident_img_metas,
                                         trident_gt_bboxes, trident_gt_labels)
