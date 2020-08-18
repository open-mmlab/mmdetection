import copy
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import build_from_cfg

from mmdet.datasets.builder import DATASETS, PIPELINES


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

    # test assertion for the num of elements in tuple fill_val
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate', level=1, fill_val=(128, 128, 128, 128))
        build_from_cfg(transform, PIPELINES)

    # test ValueError for invalid type of fill_val
    with pytest.raises(ValueError):
        transform = dict(type='Translate', level=1, fill_val=[128, 128, 128])
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of fill_val
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, fill_val=(128, -1, 256))
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid value of axis
    with pytest.raises(AssertionError):
        transform = dict(type='Translate', level=1, fill_val=128, axis='z')
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid type of max_translate_offset
    with pytest.raises(AssertionError):
        transform = dict(
            type='Translate',
            level=1,
            fill_val=128,
            max_translate_offset=(250., ))
        build_from_cfg(transform, PIPELINES)

    # construct CocoDataset as the dataset example for testing
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            with_seg=True),
    ]
    # TODO handle data_root, assume folder 'coco' is in path test/data
    data_root = osp.join(osp.dirname(__file__), '../data/coco/')
    val_data = dict(
        type='CocoDataset',
        ann_file=osp.join(data_root, 'annotations/instances_val2017.json'),
        img_prefix=osp.join(data_root, 'val2017/'),
        seg_prefix=osp.join(data_root, 'stuffthingmaps/val2017/'),
        pipeline=train_pipeline,
        test_mode=False)
    coco_dataset = build_from_cfg(val_data, DATASETS)
    # randomly sample one image and load the results according to
    # ``train_pipeline``
    results = coco_dataset.__getitem__(
        np.random.choice(range(len(coco_dataset))))

    def _check_keys(results, results_translated):
        assert len(
            set(results.keys()).difference(set(
                results_translated.keys()))) == 0
        assert len(
            set(results_translated.keys()).difference(set(
                results.keys()))) == 0

    def _pad(h, w, c, pad_val, axis=-1, dtype=np.float32):
        assert isinstance(pad_val, (int, float, tuple))
        if isinstance(pad_val, (int, float)):
            pad_val = tuple([pad_val] * c)
        assert len(pad_val) == c
        pad_data = np.stack([np.ones((h, w)) * pad_val[i] for i in range(c)],
                            axis=axis).astype(dtype)
        return pad_data

    def _check_bbox_mask(results,
                         results_translated,
                         offset,
                         axis,
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

        def _translate_bbox(bboxes, offset, axis, max_h, max_w):
            if axis == 'x':
                bboxes[:, 0::2] = bboxes[:, 0::2] + offset
            elif axis == 'y':
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
                copy.deepcopy(results[key]), offset, axis, h, w)
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
            if axis == 'x':
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
                       axis):
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
            if axis == 'x':
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
                        axis,
                        min_size=0):
        # check keys
        _check_keys(results, results_translated)
        # check image
        _check_img_seg(results, results_translated,
                       results.get('img_fields', ['img']), offset,
                       img_fill_val, axis)
        # check segmentation map
        _check_img_seg(results, results_translated,
                       results.get('seg_fields', []), offset, seg_ignore_label,
                       axis)
        # check masks and bboxes
        _check_bbox_mask(results, results_translated, offset, axis, min_size)

    # test case when level=0 (without translate aug)
    img_fill_val = (104, 116, 124)
    seg_ignore_label = 255
    transform = dict(
        type='Translate',
        level=0,
        prob=1.0,
        fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label)
    translate_module = build_from_cfg(transform, PIPELINES)
    results_wo_translate = translate_module(copy.deepcopy(results))
    check_translate(
        copy.deepcopy(results),
        results_wo_translate,
        0,
        img_fill_val,
        seg_ignore_label,
        'x',
    )

    # test case when level>0 and translate along x-axis (left shift).
    transform = dict(
        type='Translate',
        level=8,
        prob=1.0,
        fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        axis='x')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=1.0)
    check_translate(
        copy.deepcopy(results),
        results_translated,
        -offset,
        img_fill_val,
        seg_ignore_label,
        'x',
    )

    # test case when level>0 and translate along x-axis (right shift).
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=0.0)
    check_translate(
        copy.deepcopy(results),
        results_translated,
        offset,
        img_fill_val,
        seg_ignore_label,
        'x',
    )

    # test case when level>0 and translate along y-axis (top shift).
    transform = dict(
        type='Translate',
        level=10,
        prob=1.0,
        fill_val=img_fill_val,
        seg_ignore_label=seg_ignore_label,
        axis='y')
    translate_module = build_from_cfg(transform, PIPELINES)
    offset = translate_module.offset
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=1.0)
    check_translate(
        copy.deepcopy(results), results_translated, -offset, img_fill_val,
        seg_ignore_label, 'y')

    # test case when level>0 and translate along y-axis (bottom shift).
    results_translated = translate_module(
        copy.deepcopy(results), neg_offset_prob=0.0)
    check_translate(
        copy.deepcopy(results), results_translated, offset, img_fill_val,
        seg_ignore_label, 'y')
