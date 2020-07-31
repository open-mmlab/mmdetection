import numpy as np
import torch

from mmdet.core import bbox2result
from ..builder import HEADS, build_head
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class KeypointRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, output_heatmaps=False, keypoint_decoder=None, **kwargs):
        super().__init__(**kwargs)
        self.output_heatmaps = output_heatmaps
        if keypoint_decoder:
            self.keypoint_decoder = build_head(keypoint_decoder)
        else:
            assert output_heatmaps is True
            self.keypoint_decoder = None

    def init_keypoint_head(self, keypoint_roi_extractor, keypoint_head):
        self.share_roi_extractor = False
        # if keypoint_roi_extractor is not None:
        #     self.keypoint_roi_extractor = build_roi_extractor(
        # keypoint_roi_extractor)
        #     self.share_roi_extractor = False
        # else:
        #     self.share_roi_extractor = True
        #     self.keypoint_roi_extractor = self.bbox_roi_extractor
        self.keypoint_head = build_head(keypoint_head)

    def init_weights(self, pretrained):
        super().init_weights(pretrained)
        if self.with_keypoint and self.keypoint_head:
            self.keypoint_head.init_weights()

    def forward_dummy(self, x, proposals):
        outs = super().forward_dummy(x, proposals)
        # keypoints head
        if self.with_keypoint:
            pass

        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_keypoints=None,
                      gt_masks=None,
                      heatmaps=None):
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
        # assign gts and sample proposals
        sampling_results = []
        bbox_results = {'bbox_feats': []}
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
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

        if self.with_keypoint:
            keypoint_results = self._keypoint_forward_train(
                x, sampling_results, bbox_results['bbox_feats'], gt_keypoints,
                heatmaps, img_metas)
            if keypoint_results['loss_keypoint'] is not None:
                losses.update(keypoint_results['loss_keypoint'])

        return losses

    def _keypoint_forward_train(self, x, sampling_results, bbox_feats,
                                gt_keypoints, heatmaps, img_metas):
        if not self.share_roi_extractor:
            # pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            # if pos_rois.shape[0] == 0:
            #     return dict(loss_keypoint=None)
            keypoint_results = self._keypoint_forward(x)
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
                return dict(loss_keypoint=None)
            keypoint_results = self._keypoint_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        # keypoint_targets, valids = self.keypoint_head.get_target(
        #   sampling_results, gt_keypoints, self.train_cfg)
        heatmap = torch.reshape(
            torch.cat(heatmaps, dim=0), (-1, *heatmaps[0].shape))
        loss_keypoint = self.keypoint_head.loss(keypoint_results['heatmaps'],
                                                heatmap, 0)
        keypoint_results.update(
            loss_keypoint=loss_keypoint, keypoint_targets=gt_keypoints)
        return keypoint_results

    def _keypoint_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        keypoint_pred = self.keypoint_head(x)
        keypoint_results = dict(heatmaps=keypoint_pred)
        return keypoint_results

    def simple_test_keypoints(self,
                              x,
                              img_metas,
                              proposals=None,
                              rcnn_test_cfg=None,
                              rescale=False):
        """Test only keypoints without augmentation."""
        assert self.keypoint_decoder is not None
        keypoint_results = self._keypoint_forward(x)
        scale_factor = img_metas[0]['scale_factor']

        # Convert heatmaps to keypoints
        res = keypoint_results['heatmaps']
        pred = self.keypoint_decoder(res)
        keypoint_results['keypoints'] = pred.cpu().numpy()
        # Upscale keypoints to the original size
        pred[:, :, 0] /= scale_factor[0]
        pred[:, :, 1] /= scale_factor[1]
        if self.output_heatmaps:
            keypoint_results['heatmaps'] = keypoint_results['heatmaps'].cpu(
            ).numpy()
        else:
            keypoint_results.pop('heatmaps')
        return keypoint_results

    async def async_test_keypoints(self,
                                   x,
                                   img_metas,
                                   proposals=None,
                                   rcnn_test_cfg=None,
                                   rescale=False):
        """Test only keypoints without augmentation."""
        assert self.keypoint_decoder is not None
        keypoint_results = self._keypoint_forward(x)
        scale_factor = img_metas[0]['scale_factor']

        # Convert heatmaps to keypoints
        res = keypoint_results['heatmaps']
        pred = self.keypoint_decoder(res)
        keypoint_results['keypoints'] = pred.cpu().numpy()
        # Upscale keypoints to the original size
        pred[:, :, 0] /= scale_factor[0]
        pred[:, :, 1] /= scale_factor[1]
        if self.output_heatmaps:
            keypoint_results['heatmaps'] = keypoint_results['heatmaps'].cpu(
            ).numpy()
        else:
            keypoint_results.pop('heatmaps')
        return keypoint_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        if self.with_bbox:
            det_bboxes, det_labels = await self.async_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
        else:
            bbox_results = np.zeros((1, 0, 5))

        if not self.with_mask:
            segm_results = None
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))

        result = {'bbox': bbox_results, 'mask': segm_results}
        if self.with_keypoint:
            if self.keypoint_decoder is not None:
                kpts_results = self.async_test_keypoints(
                    x, img_metas, rescale=rescale)
                result.update(kpts_results)
        else:
            kpts_results = None

        return result

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, 'Bbox head must be implemented.'

        if self.with_bbox:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head.num_classes)
        else:
            bbox_results = np.zeros((1, 0, 5))

        if self.with_mask:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
        else:
            segm_results = None

        result = {'bbox': bbox_results, 'mask': segm_results}
        if self.with_keypoint:
            # if self.with_bbox:
            #     kpts_results = self.simple_test_keypoints(
            #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
            # else:
            #     kpts_results = self.simple_test_keypoints(x, img_metas,
            #         rescale=rescale)
            if self.keypoint_decoder is not None:
                kpts_results = self.simple_test_keypoints(
                    x, img_metas, rescale=rescale)
                result.update(kpts_results)
        else:
            kpts_results = None

        return result

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
