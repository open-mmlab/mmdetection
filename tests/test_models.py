# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
import os
import unittest
import sys

from common import replace_text_in_file, collect_ap


class PublicModelsTestCase(unittest.TestCase):
    coco_dir = '/tmp/data/coco'
    snapshots_dir = '/tmp/snapshots'

    @staticmethod
    def shorten_annotation(path, num_images):
        with open(path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted([item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False

        os.makedirs(cls.coco_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017.zip')):
            os.system(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017')):
            os.system(f'unzip {os.path.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, "annotations_trainval2017.zip")):
            os.system(
                f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}')
        if cls.test_on_full or not os.path.exists(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            os.system(
                    f'unzip -o {os.path.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}')

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 10

        cls.shorten_annotation(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json'), cls.shorten_to)

    def run_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.0):
        print('\n\ntesting ' + config_path, file=sys.stderr)
        name = config_path.replace('configs', '')[:-3]
        print('expected ouputs', f'tests/expected_outputs/public/{name}-{self.shorten_to}.json')
        test_dir = f'/tmp/{name}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        os.system(f'cp -r configs {test_dir}')
        target_config_path = os.path.join(test_dir, config_path)
        assert replace_text_in_file(f'{test_dir}/configs/_base_/datasets/coco_detection.py',
                                    "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        replace_text_in_file(f'{target_config_path}',
                             "data_root = 'data/coco/'",
                             f"data_root = '{self.coco_dir}/'")

        if not self.test_on_full:
            assert replace_text_in_file(f'{test_dir}/configs/_base_/datasets/coco_detection.py',
                                        "keep_ratio=True", "keep_ratio=False")
            replace_text_in_file(f'{target_config_path}',
                                 "keep_ratio=True", "keep_ratio=False")
        metrics = ' '.join(metrics)

        os.system(f'python tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval {metrics}  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{name}-{self.shorten_to}.json') as read_file:
            content = json.load(read_file)
        reference_ap = content['map']

        print(f'expected {reference_ap} vs actual {ap}')

        for expected, actual, m in zip(reference_ap, ap, metrics.split(' ')):
            if expected - thr > actual:
                raise AssertionError(f'{m}: {expected} (expected) - {thr} (threshold) > {actual}')

    def run_export_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.02, alt_ssd_export=False):
        print('\n\ntesting export ' + '(--alt_ssd_export)' if alt_ssd_export else '' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        print('expected ouputs', f'tests/expected_outputs/public/{name}-{self.shorten_to}.json')
        test_dir = f'/tmp/{name}'
        export_dir = os.path.join(test_dir, 'alt_ssd_export' if alt_ssd_export else 'export')
        log_file = os.path.join(export_dir, 'test_export.log')
        os.makedirs(export_dir, exist_ok=True)
        os.system(f'cp -r configs {test_dir}')
        target_config_path = os.path.join(test_dir, config_path)
        assert replace_text_in_file(f'{test_dir}/configs/_base_/datasets/coco_detection.py',
                                    "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")
        replace_text_in_file(f'{target_config_path}',
                             "data_root = 'data/coco/'",
                             f"data_root = '{self.coco_dir}/'")

        if not self.test_on_full:
            assert replace_text_in_file(f'{test_dir}/configs/_base_/datasets/coco_detection.py',
                                        "keep_ratio=True", "keep_ratio=False")
            replace_text_in_file(f'{target_config_path}',
                                 "keep_ratio=True", "keep_ratio=False")

        metrics = ' '.join(metrics)

        os.system(
            f'/opt/intel/openvino/bin/setupvars.sh;'
            f'python tools/export.py '
            f'{target_config_path} '
            f'{snapshot} '
            f'{export_dir} '
            f'openvino {"--alt_ssd_export" if alt_ssd_export else ""};'
            f'python tools/test_exported.py '
            f'{target_config_path} '
            f'{os.path.join(export_dir, os.path.basename(name) + ".xml")} '
            f'--out res.pkl --eval {metrics} 2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{name}-{self.shorten_to}.json') as read_file:
            content = json.load(read_file)
        reference_ap = content['map']

        print(f'expected {reference_ap} vs actual {ap}')

        for expected, actual, m in zip(reference_ap, ap, metrics.split(' ')):
            if expected - thr > actual:
                raise AssertionError(f'{m}: {expected} (expected) - {thr} (threshold) > {actual}')

    def download_if_not_yet(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = os.path.join(self.snapshots_dir, os.path.basename(url))
        if not os.path.exists(path):
            os.system(f'wget {url} -P {self.snapshots_dir}')
        return path

    def test_atss__atss_r50_fpn_1x_coco(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_dcn__faster_rcnn_r50_fpn_dconv_c3_5_1x_coco(self):
        origin_config = 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_gn__mask_rcnn_r50_fpn_gn_all_2x_coco(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_gn_ws__faster_rcnn_r50_fpn_gn_ws_all_1x_coco(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_fcos__fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_4x2_2x_coco(self):
        origin_config = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
              'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_foveabox__fovea_r50_fpn_4x4_1x_coco(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    # def test_ms_rcnn__ms_rcnn_r50_caffe_fpn_1x(self):
    #     origin_config = 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/' \
    #           'ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'
    #     self.run_test(origin_config, self.download_if_not_yet(url))
    #
    # def test_htc__htc_r50_fpn_1x(self):
    #     origin_config = 'configs/htc/htc_r50_fpn_1x.py'
    #     url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/' \
    #           'htc_r50_fpn_1x_20190408-878c1712.pth'
    #     self.run_test(origin_config, self.download_if_not_yet(url))

    def test_cascade_rcnn__cascade_mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_cascade_rcnn__cascade_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_faster_rcnn__faster_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_mask_rcnn__mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_ssd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    # Export

    def test_export_atss__atss_r50_fpn_1x_coco(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_dcn__faster_rcnn_r50_fpn_dconv_c3_5_1x_coco(self):
        origin_config = 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_gn__mask_rcnn_r50_fpn_gn_all_2x_coco(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_gn_ws__faster_rcnn_r50_fpn_gn_ws_all_1x_coco(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_fcos__fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_4x2_2x_coco(self):
        origin_config = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
              'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_foveabox__fovea_r50_fpn_4x4_1x_coco(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    # def test_export_ms_rcnn__ms_rcnn_r50_caffe_fpn_1x(self):
    #     origin_config = 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/' \
    #           'ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth'
    #     self.run_export_test(origin_config, self.download_if_not_yet(url))
    #
    # def test_export_htc__htc_r50_fpn_1x(self):
    #     origin_config = 'configs/htc/htc_r50_fpn_1x.py'
    #     url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/' \
    #           'htc_r50_fpn_1x_20190408-878c1712.pth'
    #     self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_cascade_rcnn__cascade_mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_cascade_rcnn__cascade_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_faster_rcnn__faster_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_mask_rcnn__mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_sd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_alt_ssd_export_sd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), alt_ssd_export=True)


if __name__ == '__main__':
    unittest.main()
