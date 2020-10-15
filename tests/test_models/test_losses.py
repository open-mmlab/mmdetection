import numpy as np
import pytest
import torch

from mmdet.core import BboxGIoU2D, bbox_gious
from mmdet.models import Accuracy, build_loss


def test_ce_loss():
    # use_mask and use_sigmoid cannot be true at the same time
    with pytest.raises(AssertionError):
        loss_cfg = dict(
            type='CrossEntropyLoss',
            use_mask=True,
            use_sigmoid=True,
            loss_weight=1.0)
        build_loss(loss_cfg)

    # test loss with class weights
    loss_cls_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=[0.8, 0.2],
        loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    fake_pred = torch.Tensor([[100, -100]])
    fake_label = torch.Tensor([1]).long()
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(40.))

    loss_cls_cfg = dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    loss_cls = build_loss(loss_cls_cfg)
    assert torch.allclose(loss_cls(fake_pred, fake_label), torch.tensor(200.))


def test_accuracy():
    # test for empty pred
    pred = torch.empty(0, 4)
    label = torch.empty(0)
    accuracy = Accuracy(topk=1)
    acc = accuracy(pred, label)
    assert acc.item() == 0

    pred = torch.Tensor([[0.2, 0.3, 0.6, 0.5], [0.1, 0.1, 0.2, 0.6],
                         [0.9, 0.0, 0.0, 0.1], [0.4, 0.7, 0.1, 0.1],
                         [0.0, 0.0, 0.99, 0]])
    # test for top1
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy = Accuracy(topk=1)
    acc = accuracy(pred, true_label)
    assert acc.item() == 100

    # test for top1 with score thresh=0.8
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    accuracy = Accuracy(topk=1, thresh=0.8)
    acc = accuracy(pred, true_label)
    assert acc.item() == 40

    # test for top2
    accuracy = Accuracy(topk=2)
    label = torch.Tensor([3, 2, 0, 0, 2]).long()
    acc = accuracy(pred, label)
    assert acc.item() == 100

    # test for both top1 and top2
    accuracy = Accuracy(topk=(1, 2))
    true_label = torch.Tensor([2, 3, 0, 1, 2]).long()
    acc = accuracy(pred, true_label)
    for a in acc:
        assert a.item() == 100

    # topk is larger than pred class number
    with pytest.raises(AssertionError):
        accuracy = Accuracy(topk=5)
        accuracy(pred, true_label)

    # wrong topk type
    with pytest.raises(AssertionError):
        accuracy = Accuracy(topk='wrong type')
        accuracy(pred, true_label)

    # label size is larger than required
    with pytest.raises(AssertionError):
        label = torch.Tensor([2, 3, 0, 1, 2, 0]).long()  # size mismatch
        accuracy = Accuracy()
        accuracy(pred, label)

    # wrong pred dimension
    with pytest.raises(AssertionError):
        accuracy = Accuracy()
        accuracy(pred[:, :, None], true_label)


def test_giou_loss(nb_test=100, eps=1e-7):

    def _construct_bbox(nb_bbox=None):
        img_h = int(np.random.randint(3, 1000))
        img_w = int(np.random.randint(3, 1000))
        if nb_bbox is None:
            nb_bbox = np.random.randint(1, 10)
        x1y1 = torch.rand((nb_bbox, 2))
        x2y2 = torch.max(torch.rand((nb_bbox, 2)), x1y1)
        bboxes = torch.cat((x1y1, x2y2), -1)
        bboxes[:, 0::2] *= img_w
        bboxes[:, 1::2] *= img_h
        return bboxes, nb_bbox

    # is_aligned is True, bboxes.size(-1) == 5 (include score)
    self = BboxGIoU2D()
    bboxes1, nb_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(nb_bbox)
    bboxes1 = torch.cat((bboxes1, torch.rand((nb_bbox, 1))), 1)
    bboxes2 = torch.cat((bboxes2, torch.rand((nb_bbox, 1))), 1)
    gious = self(bboxes1, bboxes2, True)
    assert gious.size() == (nb_bbox, 1), gious.size()
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    # is_aligned is True, bboxes1.size(-2) == 0
    bboxes1 = torch.empty((0, 4))
    bboxes2 = torch.empty((0, 4))
    gious = self(bboxes1, bboxes2, True)
    assert gious.size() == (0, 1), gious.size()
    assert torch.all(gious == torch.empty((0, 1)))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    # is_aligned is True, and bboxes.ndims > 2
    bboxes1, nb_bbox = _construct_bbox()
    bboxes2, _ = _construct_bbox(nb_bbox)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    # test assertion when batch dim is not the same
    with pytest.raises(AssertionError):
        self(bboxes1, bboxes2.unsqueeze(0).repeat(3, 1, 1), True)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2, True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, nb_bbox, 1)
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1, 1)
    gious = self(bboxes1, bboxes2, True)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, 2, nb_bbox, 1)

    # is_aligned is False
    bboxes1, nb_bbox1 = _construct_bbox()
    bboxes2, nb_bbox2 = _construct_bbox()
    gious = self(bboxes1, bboxes2)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (nb_bbox1, nb_bbox2)

    # is_aligned is False, and bboxes.ndims > 2
    bboxes1 = bboxes1.unsqueeze(0).repeat(2, 1, 1)
    bboxes2 = bboxes2.unsqueeze(0).repeat(2, 1, 1)
    gious = self(bboxes1, bboxes2)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (2, nb_bbox1, nb_bbox2)
    bboxes1 = bboxes1.unsqueeze(0)
    bboxes2 = bboxes2.unsqueeze(0)
    gious = self(bboxes1, bboxes2)
    assert torch.all(gious >= -1) and torch.all(gious <= 1)
    assert gious.size() == (1, 2, nb_bbox1, nb_bbox2)

    # is_aligned is False, bboxes1.size(-2) == 0
    gious = self(torch.empty(1, 2, 0, 4), bboxes2)
    assert torch.all(gious == torch.empty(1, 2, 0, bboxes2.size(-2)))
    assert torch.all(gious >= -1) and torch.all(gious <= 1)

    def _giou_loss(pred, target):
        """gious loss using bbox_gious."""
        gious = bbox_gious(pred, target, is_aligned=True, eps=eps)
        if gious.ndim >= 2 and gious.size(-1) == 1:
            gious = gious.squeeze(-1)  # [..., n, 1] -> [..., n]
        loss = 1 - gious
        return loss

    def _giou_loss_wo(pred, target):
        """gious loss without using bbox_gious."""
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps
        # IoU
        ious = overlap / union
        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps
        # GIoU
        gious = ious - (enclose_area - union) / enclose_area
        loss = 1 - gious
        return loss

    # test allclose between above two functions
    for _ in range(nb_test):
        pred, nb_bbox = _construct_bbox()
        target, _ = _construct_bbox(nb_bbox)
        giou_loss1 = _giou_loss(pred, target)
        giou_loss2 = _giou_loss_wo(pred, target)
        assert torch.allclose(giou_loss1, giou_loss2, rtol=0, atol=eps * 5)
