from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CenterNet2(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CenterNet2, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        losses = dict()
        # RPN forward and loss

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        pure_proposal_list = [proposal[0] for proposal in proposal_list]

        roi_losses = self.roi_head.forward_train(x, img_metas, pure_proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    # >>>>>>>>>>>>>>>>>>>>>>>> TO BE COMPLETED !!! <<<<<<<<<<<<<<<<
    # async def async_simple_test(self,
    #                             img,
    #                             img_meta,
    #                             proposals=None,
    #                             rescale=False):
    #     """Async test without augmentation."""
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #
    #     if proposals is None:
    #         proposal_list = await self.rpn_head.async_simple_test_rpn(
    #             x, img_meta)
    #     else:
    #         proposal_list = proposals
    #
    #     return await self.roi_head.async_simple_test(
    #         x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        pure_proposal_list = [proposal[0] for proposal in proposal_list]

        return self.roi_head.simple_test(
            x, pure_proposal_list, img_metas, rescale=rescale)

    # def aug_test(self, imgs, img_metas, rescale=False):
    #     """Test with augmentations.
    #
    #     If rescale is False, then returned bboxes and masks will fit the scale
    #     of imgs[0].
    #     """
    #     x = self.extract_feats(imgs)
    #     proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
    #     return self.roi_head.aug_test(
    #         x, proposal_list, img_metas, rescale=rescale)
