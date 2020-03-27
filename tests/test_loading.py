import os.path as osp

import numpy as np

from mmdet.datasets.pipelines import LoadMultiImagesFromMultiFiles


def test_load_multi_images_from_multi_files():
    ref_imgs_info = dict(
        filenames=['../demo/coco_test_12510.jpg', '../demo/demo.jpg'])
    img_prefix = osp.dirname(__file__)
    results = dict(
        ref_imgs_info=ref_imgs_info, img_prefix=img_prefix, img_fields=[])
    load_multi_images_from_multi_files = LoadMultiImagesFromMultiFiles(
        key_prefix='ref', to_float32=False)
    results = load_multi_images_from_multi_files(results)
    assert len(results['ref_imgs']) == 2
    assert results['ref_imgs'][0].dtype == np.uint8

    support_imgs_info = dict(filenames=[
        osp.join(osp.dirname(__file__), '../demo/coco_test_12510.jpg'),
        osp.join(osp.dirname(__file__), '../demo/demo.jpg')
    ])
    results = dict(
        support_imgs_info=support_imgs_info, img_prefix=None, img_fields=[])
    load_multi_images_from_multi_files = LoadMultiImagesFromMultiFiles(
        key_prefix='support', to_float32=True)
    results = load_multi_images_from_multi_files(results)
    assert len(results['support_imgs']) == 2
    assert results['support_imgs'][0].dtype == np.float32
