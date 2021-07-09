import torch

from ..builder import DETECTORS, build_head
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class YOLACT(SingleStageInstanceSegmentor):
    """Implementation of `YOLACT <https://arxiv.org/abs/1904.02689>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 segm_head,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLACT, self).__init__(backbone, neck, bbox_head, mask_head,
                                     train_cfg, test_cfg, init_cfg)
        self.segm_head = build_head(segm_head)

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_masks = [
            gt_mask.to_tensor(dtype=torch.bool, device=img.device)
            for gt_mask in gt_masks
        ]
        x = self.extract_feat(img)
        losses = dict()

        # bbox_head_results is a tuple
        bbox_head_preds = self.bbox_head(x)

        det_losses, positive_infos = self.bbox_head.loss(
            *bbox_head_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(det_losses)

        mask_head_inputs = (x, gt_labels, gt_masks, img_metas)

        # when no positive_infos add gt bbox
        mask_loss = self.mask_head.forward_train(
            *mask_head_inputs,
            positive_infos=positive_infos,
            gt_bboxes=gt_bboxes,
            gt_bboxes_ignore=gt_bboxes_ignore)
        # avoid loss override
        assert not set(mask_loss.keys()) & set(losses.keys())

        losses.update(mask_loss)

        segm_head_outs = self.segm_head(x[0])
        loss_segm = self.segm_head.loss(segm_head_outs, gt_masks, gt_labels)

        assert not set(loss_segm.keys()) & set(losses.keys())
        losses.update(loss_segm)

        # check NaN and Inf
        for loss_name in losses.keys():
            assert torch.isfinite(torch.stack(losses[loss_name]))\
                .all().item(), '{} becomes infinite or NaN!'\
                .format(loss_name)

        return losses

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations."""
        raise NotImplementedError(
            'YOLACT does not support test-time augmentation')
