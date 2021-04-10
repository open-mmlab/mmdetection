import mmcv
import pytest
import torch

from mmdet.core import bbox2roi
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from .utils import _dummy_bbox_sampling


def test_bbox_head_loss():
    """Tests bbox head loss when truth is empty and non-empty."""
    self = BBoxHead(in_channels=8, roi_feat_size=3)

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
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
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
    rois = bbox2roi([res.bboxes for res in sampling_results])

    bbox_targets = self.get_targets(sampling_results, gt_bboxes, gt_labels,
                                    target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, rois, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'


@pytest.mark.parametrize(['num_sample', 'num_batch'], [[2, 2], [0, 2], [0, 0]])
def test_bbox_head_get_bboxes(num_sample, num_batch):
    self = BBoxHead(reg_class_agnostic=True)

    num_class = 6
    rois = torch.rand((num_sample, 5))
    cls_score = torch.rand((num_sample, num_class))
    bbox_pred = torch.rand((num_sample, 4))
    scale_factor = 2.0
    det_bboxes, det_labels = self.get_bboxes(
        rois, cls_score, bbox_pred, None, scale_factor, rescale=True)
    if num_sample == 0:
        assert len(det_bboxes) == 0 and len(det_labels) == 0
    else:
        assert det_bboxes.shape == bbox_pred.shape
        assert det_labels.shape == cls_score.shape

    rois = torch.rand((num_batch, num_sample, 5))
    cls_score = torch.rand((num_batch, num_sample, num_class))
    bbox_pred = torch.rand((num_batch, num_sample, 4))
    det_bboxes, det_labels = self.get_bboxes(
        rois, cls_score, bbox_pred, None, scale_factor, rescale=True)
    assert len(det_bboxes) == num_batch and len(det_labels) == num_batch


def test_refine_boxes():
    """Mirrors the doctest in
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` but checks for
    multiple values of n_roi / n_img."""
    self = BBoxHead(reg_class_agnostic=True)

    test_settings = [

        # Corner case: less rois than images
        {
            'n_roi': 2,
            'n_img': 4,
            'rng': 34285940
        },

        # Corner case: no images
        {
            'n_roi': 0,
            'n_img': 0,
            'rng': 52925222
        },

        # Corner cases: few images / rois
        {
            'n_roi': 1,
            'n_img': 1,
            'rng': 1200281
        },
        {
            'n_roi': 2,
            'n_img': 1,
            'rng': 1200282
        },
        {
            'n_roi': 2,
            'n_img': 2,
            'rng': 1200283
        },
        {
            'n_roi': 1,
            'n_img': 2,
            'rng': 1200284
        },

        # Corner case: no rois few images
        {
            'n_roi': 0,
            'n_img': 1,
            'rng': 23955860
        },
        {
            'n_roi': 0,
            'n_img': 2,
            'rng': 25830516
        },

        # Corner case: no rois many images
        {
            'n_roi': 0,
            'n_img': 10,
            'rng': 671346
        },
        {
            'n_roi': 0,
            'n_img': 20,
            'rng': 699807
        },

        # Corner case: cal_similarity num rois and images
        {
            'n_roi': 20,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 10,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 5,
            'n_img': 5,
            'rng': 1200238
        },

        # ----------------------------------
        # Common case: more rois than images
        {
            'n_roi': 100,
            'n_img': 1,
            'rng': 337156
        },
        {
            'n_roi': 150,
            'n_img': 2,
            'rng': 275898
        },
        {
            'n_roi': 500,
            'n_img': 5,
            'rng': 4903221
        },
    ]

    for demokw in test_settings:
        try:
            n_roi = demokw['n_roi']
            n_img = demokw['n_img']
            rng = demokw['rng']

            print(f'Test refine_boxes case: {demokw!r}')
            tup = _demodata_refine_boxes(n_roi, n_img, rng=rng)
            rois, labels, bbox_preds, pos_is_gts, img_metas = tup
            bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
                                             pos_is_gts, img_metas)
            assert len(bboxes_list) == n_img
            assert sum(map(len, bboxes_list)) <= n_roi
            assert all(b.shape[1] == 4 for b in bboxes_list)
        except Exception:
            print(f'Test failed with demokw={demokw!r}')
            raise


def _demodata_refine_boxes(n_roi, n_img, rng=0):
    """Create random test data for the
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` method."""
    import numpy as np
    from mmdet.core.bbox.demodata import random_boxes
    from mmdet.core.bbox.demodata import ensure_rng
    try:
        import kwarray
    except ImportError:
        import pytest
        pytest.skip('kwarray is required for this test')
    scale = 512
    rng = ensure_rng(rng)
    img_metas = [{'img_shape': (scale, scale)} for _ in range(n_img)]
    # Create rois in the expected format
    roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
    if n_img == 0:
        assert n_roi == 0, 'cannot have any rois if there are no images'
        img_ids = torch.empty((0, ), dtype=torch.long)
        roi_boxes = torch.empty((0, 4), dtype=torch.float32)
    else:
        img_ids = rng.randint(0, n_img, (n_roi, ))
        img_ids = torch.from_numpy(img_ids)
    rois = torch.cat([img_ids[:, None].float(), roi_boxes], dim=1)
    # Create other args
    labels = rng.randint(0, 2, (n_roi, ))
    labels = torch.from_numpy(labels).long()
    bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
    # For each image, pretend random positive boxes are gts
    is_label_pos = (labels.numpy() > 0).astype(np.int)
    lbl_per_img = kwarray.group_items(is_label_pos, img_ids.numpy())
    pos_per_img = [sum(lbl_per_img.get(gid, [])) for gid in range(n_img)]
    # randomly generate with numpy then sort with torch
    _pos_is_gts = [
        rng.randint(0, 2, (npos, )).astype(np.uint8) for npos in pos_per_img
    ]
    pos_is_gts = [
        torch.from_numpy(p).sort(descending=True)[0] for p in _pos_is_gts
    ]
    return rois, labels, bbox_preds, pos_is_gts, img_metas
