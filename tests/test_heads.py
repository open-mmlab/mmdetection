import mmcv
import torch

from mmdet.core import build_assigner, build_sampler
from mmdet.models.anchor_heads import AnchorHead
from mmdet.models.bbox_heads import BBoxHead


def test_anchor_head_loss():
    """
    Tests anchor head loss when truth is empty and non-empty
    """
    self = AnchorHead(num_classes=4, in_channels=1)
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config({
        'assigner': {
            'type': 'MaxIoUAssigner',
            'pos_iou_thr': 0.7,
            'neg_iou_thr': 0.3,
            'min_pos_iou': 0.3,
            'ignore_iof_thr': -1
        },
        'sampler': {
            'type': 'RandomSampler',
            'num': 256,
            'pos_fraction': 0.5,
            'neg_pos_ub': -1,
            'add_gt_as_proposals': False
        },
        'allowed_border': 0,
        'pos_weight': -1,
        'debug': False
    })

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.anchor_generators))
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, cfg, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, cfg, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_bbox_head_loss():
    """
    Tests bbox head loss when truth is empty and non-empty
    """
    self = BBoxHead(in_channels=8, roi_feat_size=3)

    num_imgs = 1
    feat = torch.rand(1, 1, 3, 3)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config({'pos_weight': 1})

    def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
        """
        Create sample results that can be passed to BBoxHead.get_target
        """
        assign_config = {
            'type': 'MaxIoUAssigner',
            'pos_iou_thr': 0.5,
            'neg_iou_thr': 0.5,
            'min_pos_iou': 0.5,
            'ignore_iof_thr': -1
        }
        sampler_config = {
            'type': 'RandomSampler',
            'num': 512,
            'pos_fraction': 0.25,
            'neg_pos_ub': -1,
            'add_gt_as_proposals': True
        }
        bbox_assigner = build_assigner(assign_config)
        bbox_sampler = build_sampler(sampler_config)
        gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=feat)
            sampling_results.append(sampling_result)
        return sampling_results

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_target(sampling_results, gt_bboxes, gt_labels,
                                   target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) == 0, 'empty gt loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_target(sampling_results, gt_bboxes, gt_labels,
                                   target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'
