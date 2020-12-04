import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_bbox_coder
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SparseRCNNHead(CascadeRoIHead):
    r"""The overall head for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_"""

    def __init__(self,
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
                 bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.]),
                 train_cfg=None,
                 test_cfg=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        super(SparseRCNNHead, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.init_proposal_layers()
        self.init_proposal_weights()

    def init_weights(self):
        super(SparseRCNNHead, self).init_weights(pretrained=False)

    def init_proposal_layers(self):
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel)

    def init_proposal_weights(self):
        """Initialize the weights in proposal layers."""
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)
        # TODO: check init of self.init_proposal_features

    def forward_dummy(self, x):
        """Dummy forward function."""
        # bbox head
        outs = ()
        proposals = self._decode_init_proposals()
        proposal_feats = self.init_proposal_features.weight
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_results = self._bbox_forward(i, x, proposals,
                                                  proposal_feats)
                outs = outs + (bbox_results['cls_score'],
                               bbox_results['bbox_pred'])
        return outs

    def _bbox_forward(self, stage, x, proposals, proposal_features):
        """Box head forward function used in both training and testing."""
        N = len(proposals)
        num_proposals = proposals[0].shape[0]
        rois = bbox2roi(proposals)

        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_delta, proposal_features = bbox_head(bbox_feats,
                                                             proposal_features)

        bbox_pred = self.bbox_coder.decode(
            torch.cat(proposals, dim=0),
            bbox_delta.view(-1, 4),
            wh_ratio_clip=16 / 100000)
        bbox_pred = bbox_pred.view(N, num_proposals, -1).detach().unbind(0)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats,
            proposal_features=proposal_features)

        return bbox_results

    def _bbox_forward_train(self, stage, x, proposals, proposal_features,
                            gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        bbox_results = self._bbox_forward(stage, x, proposals,
                                          proposal_features)

        # TODO: compute target and losses
        # bbox_targets = self.bbox_head[stage].get_targets(
        #     gt_bboxes, gt_labels)
        # loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
        #                                        bbox_results['bbox_pred'], rois,
        #                                        *bbox_targets)

        # bbox_results.update(
        #     loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def _decode_init_proposals(self, img_metas=None):
        proposals = self.init_proposal_bboxes.weight.clone()
        x_center, y_center, w, h = proposals.unbind(-1)
        proposals = torch.stack(
            [(x_center - 0.5 * w), (y_center - 0.5 * h),
             (x_center + 0.5 * w), (y_center + 0.5 * h)], dim=-1)
        if img_metas is not None:
            if not isinstance(img_metas, list):
                img_metas = [img_metas]
            # TODO: training with batch_intput_shape may got higher performance
            # wh = img_metas[0]['batch_intput_shape'][:2][::-1]
            wh = img_metas[0]['img_shape'][:2][::-1]  # for align eval map
            whwh = torch.tensor(wh + wh).type_as(proposals)
            proposals = proposals * whwh
        return proposals

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
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
        proposals = self._decode_init_proposals(img_metas)
        proposals = [proposals.clone() for _ in range(num_imgs)]

        proposal_feats = self.init_proposal_features.weight
        proposal_feats = proposal_feats[None].repeat(num_imgs, 1, 1)

        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            lw = self.stage_loss_weights[i]

            if self.with_bbox or self.with_mask:
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, proposals,
                                                    proposal_feats,
                                                    gt_bboxes, gt_labels)

            # TODO: update stage losses
            # for name, value in bbox_results['loss_bbox'].items():
            #     losses[f's{i}.{name}'] = (
            #         value * lw if 'loss' in name else value)

            proposals = bbox_results['bbox_pred']
            proposal_feats = bbox_results['proposal_features']

        return losses

    def simple_test(self, x, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        # Decode initial proposals
        num_imgs = len(img_metas)
        proposals = self._decode_init_proposals(img_metas)
        proposals = [proposals.clone() for _ in range(num_imgs)]

        proposal_feats = self.init_proposal_features.weight
        proposal_feats = proposal_feats[None].repeat(num_imgs, 1, 1)

        for i in range(self.num_stages):
            self.current_stage = i
            bbox_results = self._bbox_forward(i, x, proposals, proposal_feats)

            proposals = bbox_results['bbox_pred']
            proposal_feats = bbox_results['proposal_features']

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

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
                    0, 1).topk(self.test_cfg.max_per_img, sorted=False)
                labels_per_img = labels[topk_indices]
                bbox_pred_per_img = bbox_pred[img_id].view(-1, 1, 4).repeat(
                    1, num_classes, 1).view(-1, 4)[topk_indices]
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(
                        scale_factor)
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
                det_labels.append(labels_per_img)
        else:
            raise NotImplementedError

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError
