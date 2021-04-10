from collections.abc import Sequence

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .bbox_overlaps import bbox_overlaps


def _recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def set_recall_param(proposal_nums, iou_thrs):
    """Check proposal_nums and iou_thrs and set correct format."""
    if isinstance(proposal_nums, Sequence):
        _proposal_nums = np.array(proposal_nums)
    elif isinstance(proposal_nums, int):
        _proposal_nums = np.array([proposal_nums])
    else:
        _proposal_nums = proposal_nums

    if iou_thrs is None:
        _iou_thrs = np.array([0.5])
    elif isinstance(iou_thrs, Sequence):
        _iou_thrs = np.array(iou_thrs)
    elif isinstance(iou_thrs, float):
        _iou_thrs = np.array([iou_thrs])
    else:
        _iou_thrs = iou_thrs

    return _proposal_nums, _iou_thrs


def eval_recalls(gts,
                 proposals,
                 proposal_nums=None,
                 iou_thrs=0.5,
                 logger=None):
    """Calculate recalls.

    Args:
        gts (list[ndarray]): a list of arrays of shape (n, 4)
        proposals (list[ndarray]): a list of arrays of shape (k, 4) or (k, 5)
        proposal_nums (int | Sequence[int]): Top N proposals to be evaluated.
        iou_thrs (float | Sequence[float]): IoU thresholds. Default: 0.5.
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.

    Returns:
        ndarray: recalls of different ious and proposal nums
    """

    img_num = len(gts)
    assert img_num == len(proposals)

    proposal_nums, iou_thrs = set_recall_param(proposal_nums, iou_thrs)

    all_ious = []
    for i in range(img_num):
        if proposals[i].ndim == 2 and proposals[i].shape[1] == 5:
            scores = proposals[i][:, 4]
            sort_idx = np.argsort(scores)[::-1]
            img_proposal = proposals[i][sort_idx, :]
        else:
            img_proposal = proposals[i]
        prop_num = min(img_proposal.shape[0], proposal_nums[-1])
        if gts[i] is None or gts[i].shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts[i], img_proposal[:prop_num, :4])
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)

    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls


def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        iou_thrs (ndarray or list): iou thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(iou thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmcv.utils.print_log()` for details. Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)


def plot_num_recall(recalls, proposal_nums):
    """Plot Proposal_num-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        proposal_nums(ndarray or list): same shape as `recalls`
    """
    if isinstance(proposal_nums, np.ndarray):
        _proposal_nums = proposal_nums.tolist()
    else:
        _proposal_nums = proposal_nums
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot([0] + _proposal_nums, [0] + _recalls)
    plt.xlabel('Proposal num')
    plt.ylabel('Recall')
    plt.axis([0, proposal_nums.max(), 0, 1])
    f.show()


def plot_iou_recall(recalls, iou_thrs):
    """Plot IoU-Recalls curve.

    Args:
        recalls(ndarray or list): shape (k,)
        iou_thrs(ndarray or list): same shape as `recalls`
    """
    if isinstance(iou_thrs, np.ndarray):
        _iou_thrs = iou_thrs.tolist()
    else:
        _iou_thrs = iou_thrs
    if isinstance(recalls, np.ndarray):
        _recalls = recalls.tolist()
    else:
        _recalls = recalls

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot(_iou_thrs + [1.0], _recalls + [0.])
    plt.xlabel('IoU')
    plt.ylabel('Recall')
    plt.axis([iou_thrs.min(), 1, 0, 1])
    f.show()
