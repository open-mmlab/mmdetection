import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin, dummy_pad

from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
import numpy as np

import string


@HEADS.register_module()
class StandardRoIHeadWithText(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head.
    """

    def __init__(self, text_roi_extractor, text_head, text_thr, alphabet='  ' + string.ascii_lowercase + string.digits, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_text = True
        self.init_text_head(text_roi_extractor, text_head)
        self.alphabet = alphabet
        self.text_thr = text_thr
        if self.train_cfg:
            self.text_bbox_assigner = build_assigner(self.train_cfg.text_assigner)
            self.text_bbox_sampler = build_sampler(self.train_cfg.text_sampler)
            self.area_per_symbol_thr = self.train_cfg.get('area_per_symbol_thr', 0)

    def init_text_head(self, text_roi_extractor, text_head):
        self.text_roi_extractor = build_roi_extractor(text_roi_extractor)
        self.text_head = build_head(text_head)

    def forward_dummy(self, x, proposals):
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_texts=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert len(gt_texts) == len(gt_bboxes), f'{gt_texts} {gt_bboxes}'

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            text_sampling_results = []
            for i in range(num_imgs):
                assign_result = self.text_bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.text_bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels=gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                text_sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])


        text_results = self._text_forward_train(x, text_sampling_results, gt_masks, gt_texts, img_metas)
        if text_results['loss_text'] is not None:
            losses.update(text_results)

        return losses

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _text_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, matched_gt_texts=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            text_feats = self.text_roi_extractor(
                x[:self.text_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                text_feats = self.shared_head(text_feats)
        else:
            assert bbox_feats is not None
            text_feats = bbox_feats[pos_inds]

        text_results = self.text_head.forward(text_feats, matched_gt_texts)
        if self.training:
            return dict(loss_text=text_results)
        else:
            return dict(text_results=text_results)

    def _text_forward_train(self, x, sampling_results, gt_masks, gt_texts,
                            img_metas):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            matched_gt_texts = []
            for text, res in zip(gt_texts, sampling_results):
                assigned_gt_indices = list(res.pos_assigned_gt_inds.cpu().numpy())
                matched_texts = text[assigned_gt_indices]
                assert len(matched_texts) == len(assigned_gt_indices)

                matched_gt_texts.extend(matched_texts)
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)

            areas = (pos_rois[:, 3] - pos_rois[:, 1]) * (pos_rois[:, 4] - pos_rois[:, 2])
            areas = areas.detach().cpu().numpy().reshape(-1)
            # since EOS symbol added to text, subtract it
            text_lenths = np.array([max(len(text) - 1, 1) for text in matched_gt_texts])

            area_per_symbol = areas / text_lenths
            
            # removed = [text for text, aps in zip(matched_gt_texts, area_per_symbol) if aps < self.area_per_symbol_thr]
            # if removed:
            #    print(removed)
            
            matched_gt_texts = [text if aps >= self.area_per_symbol_thr else [] for text, aps in zip(matched_gt_texts, area_per_symbol)]
            
            text_results = self._text_forward(x, pos_rois, matched_gt_texts=matched_gt_texts)
        else:
            raise NotImplementedError()

        # text_targets = self.text_head.get_targets(sampling_results, gt_texts,
        #                                           self.train_cfg)
        # pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # loss_text = self.text_head.loss(text_results['text_pred'],
        #                                 text_targets, pos_labels)

        return text_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_text(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_masks,
                         rescale=False):
        # # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if torch.onnx.is_in_onnx_export() and det_bboxes.shape[0] == 0:
            # If there are no detection there is nothing to do for a mask head.
            # But during ONNX export we should run mask head
            # for it to appear in the graph.
            # So add one zero / dummy ROI that will be mapped
            # to an Identity op in the graph.
            det_bboxes = dummy_pad(det_bboxes, (0, 0, 0, 1))

        if det_bboxes.shape[0] == 0:
            decoded_texts = torch.empty([0, 0, 0],
                                    dtype=det_bboxes.dtype,
                                    device=det_bboxes.device)
            # segm_result = torch.empty([0, 0, 0],
            #                         dtype=det_bboxes.dtype,
            #                         device=det_bboxes.device)
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            text_rois = bbox2roi([_bboxes])
            text_results = self._text_forward(x, text_rois)
            if torch.onnx.is_in_onnx_export():
                return text_results
            text_results = text_results['text_results'].permute(1, 0, 2)
            text_results = torch.nn.functional.softmax(text_results, dim=-1)
            decoded_texts = []
            for text in text_results:
                predicted_confidences, encoded = text.topk(1)
                predicted_confidences = predicted_confidences.cpu().numpy()
                encoded = encoded.cpu().numpy().reshape(-1)
                decoded = ''
                confidence = 1
                for l, c in zip(encoded, predicted_confidences):
                    confidence = confidence * c
                    if l == 1:
                        break
                    decoded = decoded + self.alphabet[l]
                
                decoded_texts.append(decoded if confidence >= self.text_thr else '')
                    
                
            # segm_result = self.mask_head.get_seg_masks(
            #     mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
            #     ori_shape, scale_factor, rescale)
        return decoded_texts

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    postprocess=True):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

 

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=False)

        det_masks = None
        if self.with_mask:
            det_masks = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=False)

        det_texts = self.simple_test_text(x, img_metas, det_bboxes, det_masks)

        if postprocess:
            bbox_results, segm_results = self.postprocess(
                det_bboxes, det_labels, det_masks, img_metas, rescale=rescale)
            return bbox_results, segm_results, det_texts
        else:
            if det_masks is None:
                return det_bboxes, det_labels
            else:
                return det_bboxes, det_labels, det_masks, det_texts

    def postprocess(self,
                    det_bboxes,
                    det_labels,
                    det_masks,
                    img_meta,
                    rescale=False):
        img_h, img_w = img_meta[0]['ori_shape'][:2]
        num_classes = self.bbox_head.num_classes
        scale_factor = img_meta[0]['scale_factor']
        if isinstance(scale_factor, float):
            scale_factor = np.asarray((scale_factor, ) * 4)

        if rescale:
            # Keep original image resolution unchanged
            # and scale bboxes and masks to it.
            if isinstance(det_bboxes, torch.Tensor):
                scale_factor = det_bboxes.new_tensor(scale_factor)
            det_bboxes[:, :4] /= scale_factor
        else:
            # Resize image to test resolution
            # and keep bboxes and masks in test scale.
            img_h = np.round(img_h * scale_factor[1]).astype(np.int32)
            img_w = np.round(img_w * scale_factor[0]).astype(np.int32)

        bbox_results = bbox2result(det_bboxes, det_labels, num_classes)
        if self.with_mask:
            segm_results = mask2result(
                det_bboxes,
                det_labels,
                det_masks,
                num_classes,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
                img_size=(img_h, img_w))
            return bbox_results, segm_results

        return bbox_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
