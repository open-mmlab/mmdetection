import string

import numpy as np
import torch

from mmdet.core import bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import dummy_pad


@HEADS.register_module()
class StandardRoIHeadWithText(StandardRoIHead):
    """Simplest base roi head including one bbox head, one mask head and one text head.
    """

    def __init__(self, text_roi_extractor, text_head, text_thr,
                 alphabet='  ' + string.ascii_lowercase + string.digits,
                 mask_text_features=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_text = True
        self.init_text_head(text_roi_extractor, text_head)
        self.alphabet = alphabet
        self.text_thr = text_thr
        self.mask_text_features = mask_text_features
        if self.train_cfg:
            self.text_bbox_assigner = build_assigner(self.train_cfg.text_assigner)
            self.text_bbox_sampler = build_sampler(self.train_cfg.text_sampler)
            self.area_per_symbol_thr = self.train_cfg.get('area_per_symbol_thr', 0)

    def init_text_head(self, text_roi_extractor, text_head):
        self.text_roi_extractor = build_roi_extractor(text_roi_extractor)
        self.text_head = build_head(text_head)

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

            gt_texts (None | list[numpy.ndarray]) : true encoded texts for each box
                used if the architecture supports a text task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        assert len(gt_texts) == len(gt_bboxes), f'{gt_texts} {gt_bboxes}'

        losses = super().forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks)

        if self.with_text:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
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

            text_results = self._text_forward_train(x, text_sampling_results, gt_texts, gt_masks)
            if text_results['loss_text'] is not None:
                losses.update(text_results)

        return losses

    def _text_forward(self, x, rois=None, pos_inds=None, bbox_feats=None, matched_gt_texts=None, det_masks=None):
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

        if self.mask_text_features and det_masks:
            hard_masks = det_masks > 0.5
            hard_masks = torch.unsqueeze(hard_masks, 1)
            hard_masks = hard_masks.repeat(1, text_feats.shape[1], 1, 1)
            text_feats = text_feats * hard_masks

        text_results = self.text_head.forward(text_feats, matched_gt_texts)
        if self.training:
            return dict(loss_text=text_results)
        else:
            return dict(text_results=text_results)

    def _text_forward_train(self, x, sampling_results, gt_texts, gt_masks):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            with torch.no_grad():
                matched_gt_texts = []
                for text, res in zip(gt_texts, sampling_results):
                    assigned_gt_indices = res.pos_assigned_gt_inds.cpu().numpy()
                    matched_texts = text[assigned_gt_indices]
                    assert len(matched_texts) == len(assigned_gt_indices)

                    matched_gt_texts.extend(matched_texts)
                if pos_rois.shape[0] == 0:
                    return dict(loss_mask=None)

                areas = (pos_rois[:, 3] - pos_rois[:, 1]) * (pos_rois[:, 4] - pos_rois[:, 2])
                areas = areas.detach().cpu().numpy().reshape(-1)
                # since EOS symbol added to text, subtract it
                text_lengths = np.array([max(len(text) - 1, 1) for text in matched_gt_texts])

                area_per_symbol = areas / text_lengths

                matched_gt_texts = [text if aps >= self.area_per_symbol_thr else
                                        [] for text, aps in zip(matched_gt_texts, area_per_symbol)]

            mask_targets = self.mask_head.get_targets(sampling_results, gt_masks, self.train_cfg)
            text_results = self._text_forward(x, pos_rois, matched_gt_texts=matched_gt_texts, det_masks=mask_targets)
        else:
            raise NotImplementedError()

        return text_results

    def simple_test_text(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_masks,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas['ori_shape']
        scale_factor = img_metas['scale_factor']
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

            confidences = torch.empty([0, 0, 0],
                                    dtype=det_bboxes.dtype,
                                    device=det_bboxes.device)

            distributions = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            text_rois = bbox2roi([_bboxes])
            text_results = self._text_forward(x, text_rois, det_masks=det_masks)
            if torch.onnx.is_in_onnx_export():
                return text_results
            text_results = text_results['text_results'].permute(1, 0, 2)
            text_results = torch.nn.functional.softmax(text_results, dim=-1)
            confidences = []
            decoded_texts = []
            distributions = []
            for text in text_results:
                predicted_confidences, encoded = text.topk(1)
                predicted_confidences = predicted_confidences.cpu().numpy()
                encoded = encoded.cpu().numpy().reshape(-1)
                decoded = ''
                confidence = 1
                for l, c in zip(encoded, predicted_confidences):
                    confidence *= c
                    if l == 1:
                        break
                    decoded += self.alphabet[l]
                confidences.append(confidence)
                assert self.alphabet[0] == self.alphabet[1] == ' '
                distribution = np.transpose(text.cpu().numpy())[2:, :len(decoded) + 1]
                distributions.append(distribution)
                decoded_texts.append(decoded if confidence >= self.text_thr else '')

        return decoded_texts, confidences, distributions

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

        det_masks = [None for _ in det_bboxes]
        if self.with_mask:
            det_masks = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=False)

        det_texts = [self.simple_test_text(x, img_metas[0], det_bboxes[0], det_masks[0])]

        if postprocess:
            results = []
            for i in range(len(det_bboxes)):
                bbox_results, segm_results = self.postprocess(det_bboxes[i], det_labels[i], det_masks[i], img_metas[i], rescale=rescale)
                results.append((bbox_results, segm_results, det_texts[i]))
            return results
        else:
            if det_masks is None or None in det_masks:
                return det_bboxes, det_labels
            else:
                return det_bboxes, det_labels, det_masks, det_texts
