import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core.mask import BitmapMasks, PolygonMasks


def construct_img(results):
    h, w = results['img_info']['height'], results['img_info']['width']
    img = np.random.uniform(0, 1, (h, w, 3)) * 255
    img = img.astype(np.uint8)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img']


def construct_ann_info(h=427, w=640, c=3):
    bboxes = np.array(
        [[222.62, 217.82, 241.81, 238.93], [50.5, 329.7, 130.23, 384.96],
         [175.47, 331.97, 254.8, 389.26]],
        dtype=np.float32)
    labels = np.array([9, 2, 2], dtype=np.int64)
    bboxes_ignore = np.array([[59., 253., 311., 337.]], dtype=np.float32)
    masks = [
        [[222.62, 217.82, 222.62, 238.93, 241.81, 238.93, 240.85, 218.78]],
        [[
            69.19, 332.17, 82.39, 330.25, 97.24, 329.7, 114.01, 331.35, 116.76,
            337.39, 119.78, 343.17, 128.03, 344.54, 128.86, 347.84, 124.18,
            350.59, 129.96, 358.01, 130.23, 366.54, 129.13, 377.81, 125.28,
            382.48, 119.78, 381.93, 117.31, 377.54, 116.21, 379.46, 114.83,
            382.21, 107.14, 383.31, 105.49, 378.36, 77.99, 377.54, 75.79,
            381.11, 69.74, 381.93, 66.72, 378.91, 65.07, 377.81, 63.15, 379.19,
            62.32, 383.31, 52.7, 384.96, 50.5, 379.46, 51.32, 375.61, 51.6,
            370.11, 51.6, 364.06, 53.52, 354.99, 56.27, 344.54, 59.57, 336.29,
            66.45, 332.72
        ]],
        [[
            175.47, 386.86, 175.87, 376.44, 177.08, 351.2, 189.1, 332.77,
            194.31, 331.97, 236.37, 332.77, 244.79, 342.39, 246.79, 346.79,
            248.39, 345.99, 251.6, 345.59, 254.8, 348.0, 254.8, 351.6, 250.0,
            352.0, 250.0, 354.81, 251.6, 358.41, 251.6, 364.42, 251.6, 370.03,
            252.8, 378.04, 252.8, 384.05, 250.8, 387.26, 246.39, 387.66,
            245.19, 386.46, 242.38, 388.86, 233.97, 389.26, 232.77, 388.06,
            232.77, 383.65, 195.91, 381.25, 195.91, 384.86, 191.1, 384.86,
            187.49, 385.26, 186.69, 382.85, 184.29, 382.45, 183.09, 387.26,
            178.68, 388.46, 176.28, 387.66
        ]]
    ]
    return dict(
        bboxes=bboxes, labels=labels, bboxes_ignore=bboxes_ignore, masks=masks)


def _load_bboxes(results):
    ann_info = results['ann_info']
    results['gt_bboxes'] = ann_info['bboxes'].copy()
    results['bbox_fields'] = ['gt_bboxes']
    gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
    if gt_bboxes_ignore is not None:
        results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
        results['bbox_fields'].append('gt_bboxes_ignore')


def _load_labels(results):
    results['gt_labels'] = results['ann_info']['labels'].copy()


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def process_polygons(polygons):
    polygons = [np.array(p) for p in polygons]
    valid_polygons = []
    for polygon in polygons:
        if len(polygon) % 2 == 0 and len(polygon) >= 6:
            valid_polygons.append(polygon)
    return valid_polygons


def _load_masks(results, poly2mask=True):
    h, w = results['img_info']['height'], results['img_info']['width']
    gt_masks = results['ann_info']['masks']
    if poly2mask:
        gt_masks = BitmapMasks([_poly2mask(mask, h, w) for mask in gt_masks],
                               h, w)
    else:
        gt_masks = PolygonMasks(
            [process_polygons(polygons) for polygons in gt_masks], h, w)
    results['gt_masks'] = gt_masks
    results['mask_fields'] = ['gt_masks']


def _construct_semantic_seg(results):
    h, w = results['img_info']['height'], results['img_info']['width']
    seg_toy = (np.random.uniform(0, 1, (h, w)) * 255).astype(np.uint8)
    results['gt_semantic_seg'] = seg_toy
    results['seg_fields'] = ['gt_semantic_seg']


def construct_toy_data():
    img_info = dict(height=427, width=640)
    ann_info = construct_ann_info(h=img_info['height'], w=img_info['width'])
    results = dict(img_info=img_info, ann_info=ann_info)
    # construct image, similar to 'LoadImageFromFile'
    construct_img(results)
    # 'LoadAnnotations' (bboxes, labels, masks, semantic_seg)
    _load_bboxes(results)
    _load_labels(results)
    _load_masks(results)
    _construct_semantic_seg(results)
    return results


results = construct_toy_data()
