import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SparseRoIHead(CascadeRoIHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.with_mask, 'Mask branch in SpaseRcnn ' \
                                   'is Not implemented yet'

    def forward_train(self,
                      x,
                      init_proposal_boxes,
                      init_proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      imgs_whwh=None):

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            all_stage_bbox_results = []
            # all proposals used for roi_align are detached
            # exclude first stage
            detach_proposal_list = [
                init_proposal_boxes[i]
                for i in range(len(init_proposal_boxes))
            ]
            object_feats = init_proposal_features
            # diffrent with naive two-stage detector, we have to
            # get forward results first then do the assigning
            all_stage_loss = {}

            for stage in range(self.num_stages):
                assert self.bbox_head[stage].reg_class_agnostic
                rois = bbox2roi(detach_proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)
                sampling_results = []
                all_stage_bbox_results.append(bbox_results)
                # used for hug match
                cls_pred_list = []
                for i in range(num_imgs):
                    batch_idx_mask = rois[:, 0] == i
                    cls_pred_list.append(
                        bbox_results['cls_score'][batch_idx_mask].detach())

                for i in range(num_imgs):
                    # TODO support ignore and Hug
                    #  matching process can be Parallel
                    assign_result = self.bbox_assigner[stage].assign(
                        detach_proposal_list[i], cls_pred_list[i],
                        gt_bboxes[i], gt_labels[i], img_metas[i])
                    sampling_result = self.bbox_sampler[stage].sample(
                        assign_result,
                        detach_proposal_list[i],
                        gt_bboxes[i],
                    )
                    sampling_results.append(sampling_result)
                bbox_targets = self.bbox_head[stage].get_targets(
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    self.train_cfg[stage],
                    True,
                )
                single_stage_loss = self.bbox_head[stage].loss(
                    bbox_results['cls_score'],
                    bbox_results['decode_bbox_pred'],
                    *bbox_targets,
                    imgs_whwh,
                )
                for key, value in single_stage_loss.items():
                    all_stage_loss[f'stage{stage}_{key}'] = value * \
                                self.stage_loss_weights[stage]
                detach_proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']

        return all_stage_loss

    # TODO support gt_bbox_ignore
    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing."""
        num_imgs = len(img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, new_object_feats = bbox_head(
            bbox_feats, object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            torch.zeros_like(rois),  # dummy arg
            bbox_pred,
            [rois.new_zeros(object_feats.size(1) // num_imgs)] * num_imgs,
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            object_feats=new_object_feats,
            decode_bbox_pred=torch.cat(proposal_list),
            detach_proposal_list=[item.detach() for item in proposal_list])
        return bbox_results

    def simple_test(self,
                    x,
                    init_proposal_boxes,
                    init_proposal_features,
                    img_metas,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        num_imgs = len(img_metas)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        object_feats = init_proposal_features
        # "ms" in variable names means multi-stage
        detach_proposal_list = [
            init_proposal_boxes[i] for i in range(len(init_proposal_boxes))
        ]
        all_stage_results = []
        for stage in range(self.num_stages):
            rois = bbox2roi(detach_proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)

            detach_proposal_list = bbox_results['detach_proposal_list']
            object_feats = bbox_results['object_feats']
            all_stage_results.append(bbox_results)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        # TODO support focal loss
        for i in range(num_imgs):
            detach_proposal_list = [
                all_stage_results[-1]['detach_proposal_list'][i]
            ]
            rois = bbox2roi(detach_proposal_list)
            cls_score = all_stage_results[-1]['cls_score']
            cls_score = cls_score.view(num_imgs, -1, cls_score.size(-1))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score[i],
                None,
                img_shapes[i],
                scale_factors,
                rescale=rescale,
                cfg=None)
            max_score, det_label = scores.max(-1)
            det_bbox = torch.cat([bboxes, max_score[:, None]], dim=-1)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        # TODO add bbox nonempty in det_bbox
        if torch.onnx.is_in_onnx_export():
            return det_bboxes, det_labels
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results
