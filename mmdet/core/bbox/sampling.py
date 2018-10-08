import numpy as np
import torch

from .geometry import bbox_overlaps


def random_choice(gallery, num):
    """Random select some elements from the gallery.

    It seems that Pytorch's implementation is slower than numpy so we use numpy
    to randperm the indices.
    """
    assert len(gallery) >= num
    if isinstance(gallery, list):
        gallery = np.array(gallery)
    cands = np.arange(len(gallery))
    np.random.shuffle(cands)
    rand_inds = cands[:num]
    if not isinstance(gallery, np.ndarray):
        rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
    return gallery[rand_inds]


def bbox_assign(proposals,
                gt_bboxes,
                gt_bboxes_ignore=None,
                gt_labels=None,
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=.0,
                crowd_thr=-1):
    """Assign a corresponding gt bbox or background to each proposal/anchor.

    Each proposals will be assigned with `-1`, `0`, or a positive integer.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If `gt_bboxes_ignore` is specified, bboxes which have iof (intersection
    over foreground) with `gt_bboxes_ignore` above `crowd_thr` will be ignored.

    Args:
        proposals (Tensor): Proposals or RPN anchors, shape (n, 4).
        gt_bboxes (Tensor): Ground truth bboxes, shape (k, 4).
        gt_bboxes_ignore (Tensor, optional): shape(m, 4).
        gt_labels (Tensor, optional): shape (k, ).
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. For RPN, it is usually set as 0.3, for Fast R-CNN,
            it is usually set as pos_iou_thr
        crowd_thr (float): IoF threshold for ignoring bboxes. Negative value
            for not ignoring any bboxes.

    Returns:
        tuple: (assigned_gt_inds, argmax_overlaps, max_overlaps), shape (n, )
    """

    # calculate overlaps between the proposals and the gt boxes
    overlaps = bbox_overlaps(proposals, gt_bboxes)
    if overlaps.numel() == 0:
        raise ValueError('No gt bbox or proposals')

    # ignore proposals according to crowd bboxes
    if (crowd_thr > 0) and (gt_bboxes_ignore is
                            not None) and (gt_bboxes_ignore.numel() > 0):
        crowd_overlaps = bbox_overlaps(proposals, gt_bboxes_ignore, mode='iof')
        crowd_max_overlaps, _ = crowd_overlaps.max(dim=1)
        crowd_bboxes_inds = torch.nonzero(
            crowd_max_overlaps > crowd_thr).long()
        if crowd_bboxes_inds.numel() > 0:
            overlaps[crowd_bboxes_inds, :] = -1

    return bbox_assign_wrt_overlaps(overlaps, gt_labels, pos_iou_thr,
                                    neg_iou_thr, min_pos_iou)


def bbox_assign_wrt_overlaps(overlaps,
                             gt_labels=None,
                             pos_iou_thr=0.5,
                             neg_iou_thr=0.5,
                             min_pos_iou=.0):
    """Assign a corresponding gt bbox or background to each proposal/anchor.

    This method assign a gt bbox to every proposal, each proposals will be
    assigned with -1, 0, or a positive number. -1 means don't care, 0 means
    negative sample, positive number is the index (1-based) of assigned gt.
    The assignment is done in following steps, the order matters:

    1. assign every anchor to -1
    2. assign proposals whose iou with all gts < neg_iou_thr to 0
    3. for each anchor, if the iou with its nearest gt >= pos_iou_thr,
    assign it to that bbox
    4. for each gt bbox, assign its nearest proposals(may be more than one)
    to itself

    Args:
        overlaps (Tensor): Overlaps between n proposals and k gt_bboxes,
            shape(n, k).
        gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum IoU for a bbox to be considered as a
            positive bbox. This argument only affects the 4th step.

    Returns:
        tuple: (assigned_gt_inds, [assigned_labels], argmax_overlaps,
            max_overlaps), shape (n, )
    """
    num_bboxes, num_gts = overlaps.size(0), overlaps.size(1)
    # 1. assign -1 by default
    assigned_gt_inds = overlaps.new(num_bboxes).long().fill_(-1)

    if overlaps.numel() == 0:
        raise ValueError('No gt bbox or proposals')

    assert overlaps.size() == (num_bboxes, num_gts)
    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    max_overlaps, argmax_overlaps = overlaps.max(dim=1)
    # for each gt, which anchor best overlaps with it
    # for each gt, the max iou of all proposals
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

    # 2. assign negative: below
    if isinstance(neg_iou_thr, float):
        assigned_gt_inds[(max_overlaps >= 0)
                         & (max_overlaps < neg_iou_thr)] = 0
    elif isinstance(neg_iou_thr, tuple):
        assert len(neg_iou_thr) == 2
        assigned_gt_inds[(max_overlaps >= neg_iou_thr[0])
                         & (max_overlaps < neg_iou_thr[1])] = 0

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_overlaps >= pos_iou_thr
    assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

    # 4. assign fg: for each gt, proposals with highest IoU
    for i in range(num_gts):
        if gt_max_overlaps[i] >= min_pos_iou:
            assigned_gt_inds[overlaps[:, i] == gt_max_overlaps[i]] = i + 1

    if gt_labels is None:
        return assigned_gt_inds, argmax_overlaps, max_overlaps
    else:
        assigned_labels = assigned_gt_inds.new(num_bboxes).fill_(0)
        pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
        return assigned_gt_inds, assigned_labels, argmax_overlaps, max_overlaps


def bbox_sampling_pos(assigned_gt_inds, num_expected, balance_sampling=True):
    """Balance sampling for positive bboxes/anchors.

    1. calculate average positive num for each gt: num_per_gt
    2. sample at most num_per_gt positives for each gt
    3. random sampling from rest anchors if not enough fg
    """
    pos_inds = torch.nonzero(assigned_gt_inds > 0)
    if pos_inds.numel() != 0:
        pos_inds = pos_inds.squeeze(1)
    if pos_inds.numel() <= num_expected:
        return pos_inds
    elif not balance_sampling:
        return random_choice(pos_inds, num_expected)
    else:
        unique_gt_inds = torch.unique(assigned_gt_inds[pos_inds].cpu())
        num_gts = len(unique_gt_inds)
        num_per_gt = int(round(num_expected / float(num_gts)) + 1)
        sampled_inds = []
        for i in unique_gt_inds:
            inds = torch.nonzero(assigned_gt_inds == i.item())
            if inds.numel() != 0:
                inds = inds.squeeze(1)
            else:
                continue
            if len(inds) > num_per_gt:
                inds = random_choice(inds, num_per_gt)
            sampled_inds.append(inds)
        sampled_inds = torch.cat(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(
                list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            extra_inds = torch.from_numpy(extra_inds).to(
                assigned_gt_inds.device).long()
            sampled_inds = torch.cat([sampled_inds, extra_inds])
        elif len(sampled_inds) > num_expected:
            sampled_inds = random_choice(sampled_inds, num_expected)
        return sampled_inds


def bbox_sampling_neg(assigned_gt_inds,
                      num_expected,
                      max_overlaps=None,
                      balance_thr=0,
                      hard_fraction=0.5):
    """Balance sampling for negative bboxes/anchors.

    Negative samples are split into 2 set: hard (balance_thr <= iou <
    neg_iou_thr) and easy(iou < balance_thr). The sampling ratio is controlled
    by `hard_fraction`.
    """
    neg_inds = torch.nonzero(assigned_gt_inds == 0)
    if neg_inds.numel() != 0:
        neg_inds = neg_inds.squeeze(1)
    if len(neg_inds) <= num_expected:
        return neg_inds
    elif balance_thr <= 0:
        # uniform sampling among all negative samples
        return random_choice(neg_inds, num_expected)
    else:
        assert max_overlaps is not None
        max_overlaps = max_overlaps.cpu().numpy()
        # balance sampling for negative samples
        neg_set = set(neg_inds.cpu().numpy())
        easy_set = set(
            np.where(
                np.logical_and(max_overlaps >= 0,
                               max_overlaps < balance_thr))[0])
        hard_set = set(np.where(max_overlaps >= balance_thr)[0])
        easy_neg_inds = list(easy_set & neg_set)
        hard_neg_inds = list(hard_set & neg_set)

        num_expected_hard = int(num_expected * hard_fraction)
        if len(hard_neg_inds) > num_expected_hard:
            sampled_hard_inds = random_choice(hard_neg_inds, num_expected_hard)
        else:
            sampled_hard_inds = np.array(hard_neg_inds, dtype=np.int)
        num_expected_easy = num_expected - len(sampled_hard_inds)
        if len(easy_neg_inds) > num_expected_easy:
            sampled_easy_inds = random_choice(easy_neg_inds, num_expected_easy)
        else:
            sampled_easy_inds = np.array(easy_neg_inds, dtype=np.int)
        sampled_inds = np.concatenate((sampled_easy_inds, sampled_hard_inds))
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(neg_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate((sampled_inds, extra_inds))
        sampled_inds = torch.from_numpy(sampled_inds).long().to(
            assigned_gt_inds.device)
        return sampled_inds


def bbox_sampling(assigned_gt_inds,
                  num_expected,
                  pos_fraction,
                  neg_pos_ub,
                  pos_balance_sampling=True,
                  max_overlaps=None,
                  neg_balance_thr=0,
                  neg_hard_fraction=0.5):
    """Sample positive and negative bboxes given assigned results.

    Args:
        assigned_gt_inds (Tensor): Assigned gt indices for each bbox.
        num_expected (int): Expected total samples (pos and neg).
        pos_fraction (float): Positive sample fraction.
        neg_pos_ub (float): Negative/Positive upper bound.
        pos_balance_sampling(bool): Whether to sample positive samples around
            each gt bbox evenly.
        max_overlaps (Tensor, optional): For each bbox, the max IoU of all gts.
            Used for negative balance sampling only.
        neg_balance_thr (float, optional): IoU threshold for simple/hard
            negative balance sampling.
        neg_hard_fraction (float, optional): Fraction of hard negative samples
            for negative balance sampling.

    Returns:
        tuple[Tensor]: positive bbox indices, negative bbox indices.
    """
    num_expected_pos = int(num_expected * pos_fraction)
    pos_inds = bbox_sampling_pos(assigned_gt_inds, num_expected_pos,
                                 pos_balance_sampling)
    # We found that sampled indices have duplicated items occasionally.
    # (mab be a bug of PyTorch)
    pos_inds = pos_inds.unique()
    num_sampled_pos = pos_inds.numel()
    num_neg_max = int(
        neg_pos_ub *
        num_sampled_pos) if num_sampled_pos > 0 else int(neg_pos_ub)
    num_expected_neg = min(num_neg_max, num_expected - num_sampled_pos)
    neg_inds = bbox_sampling_neg(assigned_gt_inds, num_expected_neg,
                                 max_overlaps, neg_balance_thr,
                                 neg_hard_fraction)
    neg_inds = neg_inds.unique()
    return pos_inds, neg_inds


def sample_bboxes(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels, cfg):
    """Sample positive and negative bboxes.

    This is a simple implementation of bbox sampling given candidates and
    ground truth bboxes, which includes 3 steps.

    1. Assign gt to each bbox.
    2. Add gt bboxes to the sampling pool (optional).
    3. Perform positive and negative sampling.

    Args:
        bboxes (Tensor): Boxes to be sampled from.
        gt_bboxes (Tensor): Ground truth bboxes.
        gt_bboxes_ignore (Tensor): Ignored ground truth bboxes. In MS COCO,
            `crowd` bboxes are considered as ignored.
        gt_labels (Tensor): Class labels of ground truth bboxes.
        cfg (dict): Sampling configs.

    Returns:
        tuple[Tensor]: pos_bboxes, neg_bboxes, pos_assigned_gt_inds,
            pos_gt_bboxes, pos_gt_labels
    """
    bboxes = bboxes[:, :4]
    assigned_gt_inds, assigned_labels, argmax_overlaps, max_overlaps = \
        bbox_assign(bboxes, gt_bboxes, gt_bboxes_ignore, gt_labels,
                    cfg.pos_iou_thr, cfg.neg_iou_thr, cfg.min_pos_iou,
                    cfg.crowd_thr)

    if cfg.add_gt_as_proposals:
        bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
        gt_assign_self = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=bboxes.device)
        assigned_gt_inds = torch.cat([gt_assign_self, assigned_gt_inds])
        assigned_labels = torch.cat([gt_labels, assigned_labels])

    pos_inds, neg_inds = bbox_sampling(
        assigned_gt_inds, cfg.roi_batch_size, cfg.pos_fraction, cfg.neg_pos_ub,
        cfg.pos_balance_sampling, max_overlaps, cfg.neg_balance_thr)

    pos_bboxes = bboxes[pos_inds]
    neg_bboxes = bboxes[neg_inds]
    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
    pos_gt_labels = assigned_labels[pos_inds]

    return (pos_bboxes, neg_bboxes, pos_assigned_gt_inds, pos_gt_bboxes,
            pos_gt_labels)
