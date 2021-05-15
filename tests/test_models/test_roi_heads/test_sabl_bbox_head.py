import mmcv
import torch

from mmdet.core import bbox2roi
from mmdet.models.roi_heads.bbox_heads import SABLHead
from .utils import _dummy_bbox_sampling


def test_sabl_bbox_head_loss():
    """Tests bbox head loss when truth is empty and non-empty."""
    self = SABLHead(
        num_classes=4,
        cls_in_channels=3,
        reg_in_channels=3,
        cls_out_channels=3,
        reg_offset_out_channels=3,
        reg_cls_out_channels=3,
        roi_feat_size=7)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config(dict(pos_weight=1))

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    rois = bbox2roi([res.bboxes for res in sampling_results])
    dummy_feats = torch.rand(num_sampled, 3, 7, 7)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox_cls',
                      0) == 0, 'empty gt bbox-cls-loss should be zero'
    assert losses.get('loss_bbox_reg',
                      0) == 0, 'empty gt bbox-reg-loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 3, 7, 7)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_bbox_cls',
                      0) > 0, 'empty gt bbox-cls-loss should be zero'
    assert losses.get('loss_bbox_reg',
                      0) > 0, 'empty gt bbox-reg-loss should be zero'
