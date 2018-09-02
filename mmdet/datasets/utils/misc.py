import mmcv

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale


def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def draw_bbox_and_segm(img, results, dataset, score_thr=0.5):
    bbox_results, segm_results = results
    hi_bboxes = []
    for cls_bboxes, cls_segms in zip(bbox_results, segm_results):
        if len(cls_bboxes) == 0:
            hi_bboxes.append(cls_bboxes)
            continue
        inds = np.where(cls_bboxes[:, -1] > score_thr)[0]
        hi_bboxes.append(cls_bboxes[inds, :])
        color_mask = np.random.random((1, 3))
        for i in inds:
            mask = maskUtils.decode(cls_segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    mmcv.draw_bboxes_with_label(np.ascontiguousarray(img), hi_bboxes, dataset)
