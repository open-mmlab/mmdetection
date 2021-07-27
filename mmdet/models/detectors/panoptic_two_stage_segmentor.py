import torch

from mmdet.core import bbox2roi, multiclass_nms
from ..builder import DETECTORS, build_head
from ..roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
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
            num_things_classes=80,
            num_stuff_classes=53):
        super(PanopticTwoStageSegmentor,
              self).__init__(backbone, neck, rpn_head, roi_head, train_cfg,
                             test_cfg, pretrained, init_cfg)
        if stuff_head is not None:
            self.stuff_head = build_head(stuff_head)
        if panoptic_fusion_head is not None:
            panoptic_cfg = test_cfg.panoptic if test_cfg is not None else None
            panoptic_fusion_head_ = panoptic_fusion_head.copy()
            panoptic_fusion_head_.update(test_cfg=panoptic_cfg)
            self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes

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

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        img_shapes = tuple(meta['ori_shape']
                           for meta in img_metas) if rescale else tuple(
                               meta['pad_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            masks = []
            for img_shape in img_shapes:
                out_shape = (0, self.num_things_classes) + img_shape[:2]
                masks.append(det_bboxes[0].new_zeros(out_shape))
            mask_pred = det_bboxes[0].new_zeros((0, 80, 28, 28))
            mask_results = dict(
                masks=masks, mask_pred=mask_pred, mask_feats=None)
            return mask_results
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        if rescale:
            if not isinstance(scale_factors[0], float):
                scale_factors = [
                    det_bboxes[0].new_tensor(scale_factor)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                _bboxes[i] * scale_factors[i] for i in range(len(_bboxes))
            ]

        mask_rois = bbox2roi(_bboxes)
        mask_results = self.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        # split batch mask prediction back to each image
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        masks = []
        for i in range(len(_bboxes)):
            det_bbox = _bboxes[i].reshape(-1, 4)
            det_label = det_labels[i].reshape(-1)

            mask_pred = mask_preds[i].sigmoid()

            box_inds = torch.arange(mask_pred.shape[0])
            mask_pred = mask_pred[box_inds, det_label][:, None]

            img_h, img_w, _ = img_shapes[i]
            mask_pred, _ = _do_paste_mask(
                mask_pred, det_bbox, img_h, img_w, skip_empty=False)
            masks.append(mask_pred)

        mask_results['masks'] = masks

        return mask_results

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        bboxes, scores = self.roi_head.simple_test_bboxes(
            x, img_metas, proposal_list, None, rescale=rescale)

        panoptic_cfg = self.test_cfg.panoptic
        # class-wise predictions
        det_bboxes = []
        det_labels = []
        for bboxe, score in zip(bboxes, scores):
            det_bbox, det_label = multiclass_nms(bboxe, score,
                                                 panoptic_cfg.score_thr,
                                                 panoptic_cfg.nms,
                                                 panoptic_cfg.max_per_img)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        mask_results = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)
        masks = mask_results['masks']

        logits = self.stuff_head.simple_test(x, img_metas, rescale)

        panoptic_results = []
        for i in range(len(det_bboxes)):
            panoptic_result = self.panoptic_fusion_head.simple_test(
                det_bboxes[i], det_labels[i], masks[i], logits[i])
            panoptic_result = panoptic_result.int().detach().cpu().numpy()
            result = dict(pan_results=panoptic_result)
            panoptic_results.append(result)
        return panoptic_results
