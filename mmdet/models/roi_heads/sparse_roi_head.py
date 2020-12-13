import torch

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SparseRoIHead(CascadeRoIHead):
    r"""The overall head for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(
            self,
            num_stages,
            stage_loss_weights,
            num_proposals=100,
            proposal_feature_channel=256,
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='DIIHead',
                num_classes=80,
                num_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                hidden_channels=256,
                dropout=0.0,
                roi_feat_size=7,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_cls=dict(),  # TODO
                loss_bbox=dict(),  # TODO
                loss_iou=dict()),  # TODO
            train_cfg=None,
            test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        super(SparseRoIHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def _bbox_forward(self, stage, x, rois, object_feats, img_metas):
        """Box head forward function used in both training and testing."""
        num_imgs = len(img_metas)

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats = bbox_head(bbox_feats,
                                                       object_feats)
        proposal_list = self.bbox_head[stage].refine_bboxes(
            rois,
            torch.ones_like(rois),  # dummy arg
            bbox_pred.view(-1, bbox_pred.size(-1)),
            [rois.new_zeros(object_feats.size(1))] * num_imgs,
            img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            decode_bbox_pred=torch.cat(proposal_list),
            roi_features=bbox_feats,
            object_feats=object_feats,
            # detach then used in label assign
            detach_cls_score_list=[cls_score[i] for i in range(num_imgs)],
            detach_proposal_list=[item.detach() for item in proposal_list],
        )

        return bbox_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
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
        # Decode initial proposals
        num_imgs = len(img_metas)
        num_poposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh[:, None, :].expand(num_imgs, num_poposals, 4)
        all_stage_bbox_results = []
        detach_proposal_list = [
            proposal_boxes[i] for i in range(len(proposal_boxes))
        ]
        object_feats = proposal_features
        # This is diffrent with naive two-stage detector, we
        # have to get forward results first then do the assigning
        all_stage_loss = {}
        for stage in range(self.num_stages):
            rois = bbox2roi(detach_proposal_list)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)

            all_stage_bbox_results.append(bbox_results)

            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            for i in range(num_imgs):
                assign_result = self.bbox_assigner[stage].assign(
                    detach_proposal_list[i], cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
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
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                                    self.stage_loss_weights[stage]
            detach_proposal_list = bbox_results['detach_proposal_list']
            object_feats = bbox_results['object_feats']

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_feats,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposals = [proposal_boxes[i] for i in range(num_imgs)]
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            rois = bbox2roi(proposals)
            bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                              img_metas)
            object_feats = bbox_results['object_feats']
            cls_score = bbox_results['cls_score']

        detach_proposal_list = bbox_results['detach_proposal_list']
        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            labels = torch.arange(num_classes)[None].repeat(
                self.num_proposals, 1).flatten(0, 1).type_as(proposal_feats)
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=False)
                labels_per_img = labels[topk_indices]
                bbox_pred_per_img = detach_proposal_list[img_id].view(
                    -1, 1, 4).repeat(1, num_classes, 1).view(-1,
                                                             4)[topk_indices]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(
                        scale_factor)
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]],
                              dim=1))
                det_labels.append(labels_per_img)
        else:
            for img_id in range(num_imgs):
                bboxes = detach_proposal_list[img_id]
                scores = cls_score[img_id].softmax(-1)
                max_score, det_label = scores.max(-1)
                det_score, topk_indices = max_score.topk(
                    self.test_cfg.max_per_img, sorted=False)
                det_label = det_label[topk_indices]
                bboxes = bboxes[topk_indices]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bboxes /= bboxes.new_tensor(scale_factor)
                det_bbox = torch.cat([bboxes, max_score[:, None]], dim=-1)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError

    def forward_dummy(
        self,
        x,
        proposal_boxes,
        proposal_features,
        img_metas,
    ):
        """Dummy forward function."""
        all_stage_bbox_results = []
        detach_proposal_list = [
            proposal_boxes[i] for i in range(len(proposal_boxes))
        ]
        object_feats = proposal_features
        if self.with_bbox:
            for stage in range(self.num_stages):
                rois = bbox2roi(detach_proposal_list)
                bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                                  img_metas)

                all_stage_bbox_results.append(bbox_results)
                detach_proposal_list = bbox_results['detach_proposal_list']
                object_feats = bbox_results['object_feats']
        return all_stage_bbox_results
