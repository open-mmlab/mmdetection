import torch

from mmdet.core import multiclass_nms
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class PanopticTwoStageSegmentor(TwoStageDetector):

    def __init__(
            self,
            backbone,
            neck=None,
            rpn_head=None,
            roi_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None,
            # for panoptic segmentation
            stuff_head=None,
            panoptic_fusion_head=None,
            num_things=80,
            num_stuff=53):
        super(PanopticTwoStageSegmentor,
              self).__init__(backbone, neck, rpn_head, roi_head, train_cfg,
                             test_cfg, pretrained, init_cfg)
        if stuff_head is not None:
            self.stuff_head = build_head(stuff_head)
        if panoptic_fusion_head is not None:
            self.panoptic_fusion_head = build_head(panoptic_fusion_head)

        self.num_things = num_things
        self.num_stuff = num_stuff
        self.num_classes = num_stuff + num_things

    @property
    def with_stuff_head(self):
        return hasattr(self, 'stuff_head') and self.stuff_head is not None

    @property
    def with_panoptic_fusion_head(self):
        return hasattr(self, 'panoptic_fusion_heads') and \
               self.panoptic_fusion_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        raise NotImplementedError(
            f'`forward_dummy` is not implemented in {self.__class__.__name__}')

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        stuff_loss = self.stuff_head.forward_train(x, gt_semantic_seg)
        losses.update(stuff_loss)

        return losses

    def simple_test_mask(self, x, img_metas, det_bboxes, rescale=False):
        """Simple test for mask head without augmentation."""
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(len(det_bboxes))]
            return segm_results

        # The length of proposals of different batches may be different.
        # In order to form a batch, a padding operation is required.
        if isinstance(det_bboxes, list):
            # padding to form a batch
            max_size = max([bboxes.size(0) for bboxes in det_bboxes])
            for i, bbox in enumerate(det_bboxes):
                supplement_bbox = bbox.new_full(
                    (max_size - bbox.size(0), bbox.size(1)), 0)
                det_bboxes[i] = torch.cat((supplement_bbox, bbox), dim=0)
            det_bboxes = torch.stack(det_bboxes, dim=0)

        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = det_bboxes.new_tensor(scale_factors)
            det_bboxes = det_bboxes * scale_factors.unsqueeze(1)

        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self.roi_head._mask_forward(x, mask_rois)

        return mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        bboxes, scores = self.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, None, rescale=rescale)

        # class-wise predictions
        det_bboxes, det_labels = multiclass_nms(
            bboxes[0], scores[0], self.test_cfg.panoptic.score_thr,
            self.test_cfg.panoptic.nms, self.test_cfg.panoptic.max_per_img)

        mask_results = self.simple_test_mask(
            x, img_metas, det_bboxes, rescale=rescale)
        mask_preds = mask_results['mask_pred']

        logits = self.stuff_head.simple_test(x, img_metas, rescale)

        pano_results = self.panoptic_fusion_head.simple_test(
            img_metas, det_bboxes, det_labels, mask_preds, logits)
        return [{'pano_results': pano_results}]
