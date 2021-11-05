import argparse
import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv import Config, DictAction
from mmcv.ops import nms

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.CLASSES)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        labels = ann['labels']
        analyze_per_img_dets(confusion_matrix, gt_bboxes, labels, res_bboxes,
                             score_thr, tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gt_bboxes,
                         gt_labels,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    true_positives = np.zeros_like(gt_labels)
    for det_label, det_bboxes in enumerate(result):
        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes[:, :4],
                det_bboxes[:, -1],
                nms_iou_thr,
                score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma'):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    confusion_matrix = \
        confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{}%'.format(int(confusion_matrix[i, j])),
                ha='center',
                va='center',
                color='w',
                size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = mmcv.load(args.prediction_path)
    assert isinstance(results, list)
    if isinstance(results[0], list):
        pass
    elif isinstance(results[0], tuple):
        results = [result[0] for result in results]
    else:
        raise TypeError('invalid type of prediction results')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    dataset = build_dataset(cfg.data.test)

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    plot_confusion_matrix(
        confusion_matrix,
        dataset.CLASSES + ('background', ),
        save_dir=args.save_dir,
        show=args.show)


if __name__ == '__main__':
    main()
