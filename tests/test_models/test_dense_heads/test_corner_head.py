import torch

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.models.dense_heads import CornerHead


def test_corner_head_loss():
    """Tests corner head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    self = CornerHead(num_classes=4, in_channels=1)

    # Corner head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // 4, s // 4) for _ in range(self.num_feat_levels)
    ]
    tl_heats, br_heats, tl_embs, br_embs, tl_offs, br_offs = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                                br_offs, gt_bboxes, gt_labels, img_metas,
                                gt_bboxes_ignore)
    empty_det_loss = sum(empty_gt_losses['det_loss'])
    empty_push_loss = sum(empty_gt_losses['push_loss'])
    empty_pull_loss = sum(empty_gt_losses['pull_loss'])
    empty_off_loss = sum(empty_gt_losses['off_loss'])
    assert empty_det_loss.item() > 0, 'det loss should be non-zero'
    assert empty_push_loss.item() == 0, (
        'there should be no push loss when there are no true boxes')
    assert empty_pull_loss.item() == 0, (
        'there should be no pull loss when there are no true boxes')
    assert empty_off_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                              br_offs, gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore)
    onegt_det_loss = sum(one_gt_losses['det_loss'])
    onegt_push_loss = sum(one_gt_losses['push_loss'])
    onegt_pull_loss = sum(one_gt_losses['pull_loss'])
    onegt_off_loss = sum(one_gt_losses['off_loss'])
    assert onegt_det_loss.item() > 0, 'det loss should be non-zero'
    assert onegt_push_loss.item() == 0, (
        'there should be no push loss when there are only one true box')
    assert onegt_pull_loss.item() > 0, 'pull loss should be non-zero'
    assert onegt_off_loss.item() > 0, 'off loss should be non-zero'

    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874],
                      [123.6667, 123.8757, 138.6326, 251.8874]]),
    ]
    gt_labels = [torch.LongTensor([2, 3])]

    # equalize the corners' embedding value of different objects to make the
    # push_loss larger than 0
    gt_bboxes_ind = (gt_bboxes[0] // 4).int().tolist()
    for tl_emb_feat, br_emb_feat in zip(tl_embs, br_embs):
        tl_emb_feat[:, :, gt_bboxes_ind[0][1],
                    gt_bboxes_ind[0][0]] = tl_emb_feat[:, :,
                                                       gt_bboxes_ind[1][1],
                                                       gt_bboxes_ind[1][0]]
        br_emb_feat[:, :, gt_bboxes_ind[0][3],
                    gt_bboxes_ind[0][2]] = br_emb_feat[:, :,
                                                       gt_bboxes_ind[1][3],
                                                       gt_bboxes_ind[1][2]]

    two_gt_losses = self.loss(tl_heats, br_heats, tl_embs, br_embs, tl_offs,
                              br_offs, gt_bboxes, gt_labels, img_metas,
                              gt_bboxes_ignore)
    twogt_det_loss = sum(two_gt_losses['det_loss'])
    twogt_push_loss = sum(two_gt_losses['push_loss'])
    twogt_pull_loss = sum(two_gt_losses['pull_loss'])
    twogt_off_loss = sum(two_gt_losses['off_loss'])
    assert twogt_det_loss.item() > 0, 'det loss should be non-zero'
    assert twogt_push_loss.item() > 0, 'push loss should be non-zero'
    assert twogt_pull_loss.item() > 0, 'pull loss should be non-zero'
    assert twogt_off_loss.item() > 0, 'off loss should be non-zero'


def test_corner_head_encode_and_decode_heatmap():
    """Tests corner head generating and decoding the heatmap."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'border': (0, 0, 0, 0)
    }]

    gt_bboxes = [
        torch.Tensor([[10, 20, 200, 240], [40, 50, 100, 200],
                      [10, 20, 200, 240]])
    ]
    gt_labels = [torch.LongTensor([1, 1, 2])]

    self = CornerHead(num_classes=4, in_channels=1, corner_emb_channels=1)

    feat = [
        torch.rand(1, 1, s // 4, s // 4) for _ in range(self.num_feat_levels)
    ]

    targets = self.get_targets(
        gt_bboxes,
        gt_labels,
        feat[0].shape,
        img_metas[0]['pad_shape'],
        with_corner_emb=self.with_corner_emb)

    gt_tl_heatmap = targets['topleft_heatmap']
    gt_br_heatmap = targets['bottomright_heatmap']
    gt_tl_offset = targets['topleft_offset']
    gt_br_offset = targets['bottomright_offset']
    embedding = targets['corner_embedding']
    [top, left], [bottom, right] = embedding[0][0]
    gt_tl_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_br_embedding_heatmap = torch.zeros([1, 1, s // 4, s // 4])
    gt_tl_embedding_heatmap[0, 0, top, left] = 1
    gt_br_embedding_heatmap[0, 0, bottom, right] = 1

    batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(
        tl_heat=gt_tl_heatmap,
        br_heat=gt_br_heatmap,
        tl_off=gt_tl_offset,
        br_off=gt_br_offset,
        tl_emb=gt_tl_embedding_heatmap,
        br_emb=gt_br_embedding_heatmap,
        img_meta=img_metas[0],
        k=100,
        kernel=3,
        distance_threshold=0.5)

    bboxes = batch_bboxes.view(-1, 4)
    scores = batch_scores.view(-1, 1)
    clses = batch_clses.view(-1, 1)

    idx = scores.argsort(dim=0, descending=True)
    bboxes = bboxes[idx].view(-1, 4)
    scores = scores[idx].view(-1)
    clses = clses[idx].view(-1)

    valid_bboxes = bboxes[torch.where(scores > 0.05)]
    valid_labels = clses[torch.where(scores > 0.05)]
    max_coordinate = valid_bboxes.max()
    offsets = valid_labels.to(valid_bboxes) * (max_coordinate + 1)
    gt_offsets = gt_labels[0].to(gt_bboxes[0]) * (max_coordinate + 1)

    offset_bboxes = valid_bboxes + offsets[:, None]
    offset_gtbboxes = gt_bboxes[0] + gt_offsets[:, None]

    iou_matrix = bbox_overlaps(offset_bboxes.numpy(), offset_gtbboxes.numpy())
    assert (iou_matrix == 1).sum() == 3
