import copy

import numpy as np
import pycocotools.mask as maskUtils
import pytest
from mmcv.utils import build_from_cfg

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES


def _check_keys(results, results_translated):
    assert len(set(results.keys()).difference(set(
        results_translated.keys()))) == 0
    assert len(set(results_translated.keys()).difference(set(
        results.keys()))) == 0


def _pad(h, w, c, pad_val, axis=-1, dtype=np.float32):
    assert isinstance(pad_val, (int, float, tuple))
    if isinstance(pad_val, (int, float)):
        pad_val = tuple([pad_val] * c)
    assert len(pad_val) == c
    pad_data = np.stack([np.ones((h, w)) * pad_val[i] for i in range(c)],
                        axis=axis).astype(dtype)
    return pad_data


def _construct_img(results):
    h, w = results['img_info']['height'], results['img_info']['width']
    img = np.random.uniform(0, 1, (h, w, 3)) * 255
    img = img.astype(np.uint8)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['img_fields'] = ['img']


def _construct_ann_info(h=427, w=640, c=3):
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


def _process_polygons(polygons):
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
            [_process_polygons(polygons) for polygons in gt_masks], h, w)
    results['gt_masks'] = gt_masks
    results['mask_fields'] = ['gt_masks']


def _construct_semantic_seg(results):
    h, w = results['img_info']['height'], results['img_info']['width']
    seg_toy = (np.random.uniform(0, 1, (h, w)) * 255).astype(np.uint8)
    results['gt_semantic_seg'] = seg_toy
    results['seg_fields'] = ['gt_semantic_seg']


def construct_toy_data(poly2mask=True):
    img_info = dict(height=427, width=640)
    ann_info = _construct_ann_info(h=img_info['height'], w=img_info['width'])
    results = dict(img_info=img_info, ann_info=ann_info)
    # construct image, similar to 'LoadImageFromFile'
    _construct_img(results)
    # 'LoadAnnotations' (bboxes, labels, masks, semantic_seg)
    _load_bboxes(results)
    _load_labels(results)
    _load_masks(results, poly2mask)
    _construct_semantic_seg(results)
    return results


def test_translate():
    # test assertion for invalid value of level
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=-1)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of level
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=[1])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, prob=-0.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for the num of elements in tuple img_fill_val
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', level=1, img_fill_val=(128, 128, 128, 128))
        build_from_cfg(transform, PIPELINES)

    # test ValueError for invalid type of img_fill_val
    with pytest.raises(ValueError):
        transform = dict(
            type='Translate', level=1, img_fill_val=[128, 128, 128])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of img_fill_val
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', level=1, img_fill_val=(128, -1, 256))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of direction
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', level=1, img_fill_val=128, direction='diagonal')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of max_translate_offset
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate',
            level=1,
            img_fill_val=128,
            max_translate_offset=(250., ))
        build_from_cfg(transform, PIPELINES)

    # construct toy data example for unit test
    results = construct_toy_data()

    def _check_bbox_mask(results,
                         results_translated,
                         offset,
                         direction,
                         min_size=0.):
        # The key correspondence from bboxes to labels and masks.
        bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

        def _translate_bbox(bboxes, offset, direction, max_h, max_w):
            if direction == 'horizontal':
                bboxes[:, 0::2] = bboxes[:, 0::2] + offset
            elif direction == 'vertical':
                bboxes[:, 1::2] = bboxes[:, 1::2] + offset
            else:
                raise ValueError
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, max_w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, max_h)
            return bboxes

        h, w, c = results_translated['img'].shape
        for key in results_translated.get('bbox_fields', []):
            label_key, mask_key = bbox2label[key], bbox2mask[key]
            # check length of key
            if label_key in results:
                assert len(results_translated[key]) == len(
                    results_translated[label_key])
            if mask_key in results:
                assert len(results_translated[key]) == len(
                    results_translated[mask_key])
            # construct gt_bboxes
            gt_bboxes = _translate_bbox(
                copy.deepcopy(results[key]), offset, direction, h, w)
            valid_inds = (gt_bboxes[:, 2] - gt_bboxes[:, 0] > min_size) & (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] > min_size)
            gt_bboxes = gt_bboxes[valid_inds]
            # check bbox
            assert np.equal(gt_bboxes, results_translated[key]).all()

            # construct gt_masks
            if mask_key not in results:
                # e.g. 'gt_masks_ignore'
                continue
            masks, masks_translated = results[mask_key].to_ndarray(
            ), results_translated[mask_key].to_ndarray()
            assert masks.dtype == masks_translated.dtype
            if direction == 'horizontal':
                masks_pad = _pad(
                    h,
                    abs(offset),
                    masks.shape[0],
                    0,
                    axis=0,
                    dtype=masks.dtype)
                if offset <= 0:
                    # left shift
                    gt_masks = np.concatenate(
                        (masks[:, :, -offset:], masks_pad), axis=-1)
                else:
                    # right shift
                    gt_masks = np.concatenate(
                        (masks_pad, masks[:, :, :-offset]), axis=-1)
            else:
                masks_pad = _pad(
                    abs(offset),
                    w,
                    masks.shape[0],
                    0,
                    axis=0,
                    dtype=masks.dtype)
                if offset <= 0:
                    # top shift
                    gt_masks = np.concatenate(
                        (masks[:, -offset:, :], masks_pad), axis=1)
                else:
                    # bottom shift
                    gt_masks = np.concatenate(
                        (masks_pad, masks[:, :-offset, :]), axis=1)
            gt_masks = gt_masks[valid_inds]
            # check masks
            assert np.equal(gt_masks, masks_translated).all()

    def _check_img_seg(results, results_translated, keys, offset, fill_val,
                       direction):
        for key in keys:
            assert isinstance(results_translated[key], type(results[key]))
            # assert type(results[key]) == type(results_translated[key])
            data, data_translated = results[key], results_translated[key]
            if 'mask' in key:
                data, data_translated = data.to_ndarray(
                ), data_translated.to_ndarray()
            assert data.dtype == data_translated.dtype
            if 'img' in key:
                data, data_translated = data.transpose(
                    (2, 0, 1)), data_translated.transpose((2, 0, 1))
            elif 'seg' in key:
                data, data_translated = data[None, :, :], data_translated[
                    None, :, :]
            c, h, w = data.shape
            if direction == 'horizontal':
                data_pad = _pad(
                    h, abs(offset), c, fill_val, axis=0, dtype=data.dtype)
                if offset <= 0:
                    # left shift
                    data_gt = np.concatenate((data[:, :, -offset:], data_pad),
                                             axis=-1)
                else:
                    # right shift
                    data_gt = np.concatenate((data_pad, data[:, :, :-offset]),
                                             axis=-1)
            else:
                data_pad = _pad(
                    abs(offset), w, c, fill_val, axis=0, dtype=data.dtype)
                if offset <= 0:
                    # top shift
                    data_gt = np.concatenate((data[:, -offset:, :], data_pad),
                                             axis=1)
                else:
                    # bottom shift
                    data_gt = np.concatenate((data_pad, data[:, :-offset, :]),
                                             axis=1)
            if 'mask' in key:
                # TODO assertion here. ``data_translated`` must be a subset
                # (or equal) of ``data_gt``
                pass
            else:
                assert np.equal(data_gt, data_translated).all()

    def check_translate(results,
                        results_translated,
                        offset,
                        img_fill_val,
                        seg_ignore_label,
                        direction,
                        min_size=0):
        # check keys
        _check_keys(results, results_translated)
        # check image
        _check_img_seg(results, results_translated,
                       results.get('img_fields', ['img']), offset,
                       img_fill_val, direction)
        # check segmentation map
        _check_img_seg(results, results_translated,
                       results.get('seg_fields', []), offset, seg_ignore_label,
                       direction)
        # check masks and bboxes
        _check_bbox_mask(results, results_translated, offset, direction,
                         min_size)

    # test case when level=0 (without translate aug)
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(
        type='Translate',
        level=0,
        prob=1.0,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    results_wo_translate = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results),
        results_wo_translate,
        0,
        img_fill_val,
        seg_ignore_label,
        'horizontal',
    )

    # test case when level>0 and translate horizontally (left shift).
    transform = dict(
        type='Translate',
        level=8,
        prob=1.0,
        img_fill_val=img_fill_val,
        random_negative_prob=1.0,
        seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results),
        results_translated,
        -offset,
        img_fill_val,
        seg_ignore_label,
        'horizontal',
    )

    # test case when level>0 and translate horizontally (right shift).
    translate_module.random_negative_prob = 0.0
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results),
        results_translated,
        offset,
        img_fill_val,
        seg_ignore_label,
        'horizontal',
    )

    # test case when level>0 and translate vertically (top shift).
    transform = dict(
        type='Translate',
        level=10,
        prob=1.0,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        random_negative_prob=1.0,
        direction='vertical')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results), results_translated, -offset, img_fill_val,
        seg_ignore_label, 'vertical')

    # test case when level>0 and translate vertically (bottom shift).
    translate_module.random_negative_prob = 0.0
    results_translated = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results), results_translated, offset, img_fill_val,
        seg_ignore_label, 'vertical')

    # test case when no translation is called (prob<=0)
    transform = dict(
        type='Translate',
        level=8,
        prob=0.0,
        img_fill_val=img_fill_val,
        random_negative_prob=0.0,
        seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    results_translated = translate_module(copy.deepcopy(results))

    # test translate vertically with PolygonMasks (top shift)
    results = construct_toy_data(False)
    transform = dict(
        type='Translate',
        level=10,
        prob=1.0,
        img_fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        direction='vertical')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    translate_module.random_negative_prob = 1.0
    results_translated = translate_module(copy.deepcopy(results))

    def _translated_gt(masks, direction, offset, out_shape):
        translated_masks = []
        for poly_per_obj in masks:
            translated_poly_per_obj = []
            for p in poly_per_obj:
                p = p.copy()
                if direction == 'horizontal':
                    p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                elif direction == 'vertical':
                    p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                if PolygonMasks([[p]], *out_shape).areas[0] > 0:
                    # filter invalid (area=0)
                    translated_poly_per_obj.append(p)
            if len(translated_poly_per_obj):
                translated_masks.append(translated_poly_per_obj)
        translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    h, w = results['img_shape'][:2]
    for key in results.get('mask_fields', []):
        masks = results[key]
        translated_gt = _translated_gt(masks, 'vertical', -offset, (h, w))
        assert np.equal(results_translated[key].to_ndarray(),
                        translated_gt.to_ndarray()).all()

    # test translate horizontally with PolygonMasks (right shift)
    results = construct_toy_data(False)
    transform = dict(
        type='Translate',
        level=8,
        prob=1.0,
        img_fill_val=img_fill_val,
        random_negative_prob=0.0,
        seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(copy.deepcopy(results))
    h, w = results['img_shape'][:2]
    for key in results.get('mask_fields', []):
        masks = results[key]
        translated_gt = _translated_gt(masks, 'horizontal', offset, (h, w))
        assert np.equal(results_translated[key].to_ndarray(),
                        translated_gt.to_ndarray()).all()

    # test AutoAugment equipped with Translate
    policies = [[dict(type='Translate', level=10, prob=1.)]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))

    policies = [[
        dict(type='Translate', level=10, prob=1.),
        dict(
            type='Translate',
            level=8,
            img_fill_val=img_fill_val,
            direction='vertical')
    ]]
    autoaug = dict(type='AutoAugment', policies=policies)
    autoaug_module = build_from_cfg(autoaug, PIPELINES)
    autoaug_module(copy.deepcopy(results))
