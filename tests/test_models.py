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
    coco_dir = '/tmp/coco'
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
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017.zip')):
            os.system(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, 'val2017')):
            os.system(f'unzip {os.path.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, "annotations_trainval2017.zip")):
            os.system(
                f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}')
        if not os.path.exists(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            os.system(
                f'unzip {os.path.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}')

        cls.shorten_annotation(os.path.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               100)

    def run_test(self, config_path, snapshot, metrics=('bbox', )):
        print('\n\ntesting ' + config_path, file=sys.stderr)
        name = config_path.replace('configs', '')[:-3]
        test_dir = f'/tmp/{name}'
        log_file = os.path.join(test_dir, 'log.txt')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = os.path.join(test_dir, 'config.py')
        os.system(f'cp {config_path} {target_config_path}')
        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")
        metrics = ' '.join(metrics)

        os.system(f'python tools/test.py '
                  f'{target_config_path} '
                  f'{snapshot} '
                  f'--out {test_dir}/res.pkl --eval {metrics}  2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{name}.json') as read_file:
            content = json.load(read_file)
        reference_ap = content['map']
        if isinstance(reference_ap, float):
            reference_ap = [reference_ap, ]

        self.assertListEqual(reference_ap, ap)

    def run_export_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.01):
        print('\n\ntesting export ' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = f'/tmp/{name}'
        export_dir = os.path.join(test_dir, "export")
        log_file = os.path.join(export_dir, 'test_export.log')
        os.makedirs(export_dir, exist_ok=True)
        target_config_path = os.path.join(export_dir, os.path.basename(config_path))
        os.system(f'cp {config_path} {target_config_path}')
        assert replace_text_in_file(target_config_path, "data_root = 'data/coco/'",
                                    f"data_root = '{self.coco_dir}/'")

        metrics = ' '.join(metrics)

        os.system(
            f'/opt/intel/openvino/bin/setupvars.sh;'
            f'python tools/export.py '
            f'{target_config_path} '
            f'{snapshot} '
            f'{export_dir} '
            f'openvino ;'
            f'python tools/test_exported.py '
            f'{target_config_path} '
            f'{os.path.join(export_dir, os.path.basename(name) + ".xml")} '
            f'--out res.pkl --eval {metrics} 2>&1 | tee {log_file}')

        ap = collect_ap(log_file)

        with open(f'tests/expected_outputs/public/{name}.json') as read_file:
            content = json.load(read_file)
        reference_ap = content['map']
        if isinstance(reference_ap, float):
            reference_ap = [reference_ap, ]

        reference_ap = [ap - thr for ap in reference_ap]

        for expected, actual in zip(reference_ap, ap):
            self.assertLess(expected, actual)

    def download_if_not_yet(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = os.path.join(self.snapshots_dir, os.path.basename(url))
        if not os.path.exists(path):
            os.system(f'wget --no-verbose {url} -P {self.snapshots_dir}')
        return path

    def test_atss__atss_r50_fpn_1x(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'models/atss/atss_r50_fpn_1x_20200113-a7aa251e.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_dcn_faster_rcnn_dconv_c3_c5_r50_fpn_1x(self):
        origin_config = 'configs/dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_gn__mask_rcnn_r50_fpn_gn_2x(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn_2x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'gn/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_gn_ws__faster_rcnn_r50_fpn_gn_ws_1x(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'ws/faster_rcnn_r50_fpn_gn_ws_1x_20190418-935d00b6.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_fcos__fcos_r50_caffe_fpn_gn_1x_4gpu(self):
        origin_config = 'configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_20190516-9f253a93.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_foveabox__fovea_r50_fpn_4gpu_1x(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4gpu_1x.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/' \
              'foveabox/fovea_r50_fpn_4gpu_1x_20190905-3b185a5d.pth'
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
    #
    def test_cascade_mask_rcnn_r50_fpn_1x(self):
        origin_config = 'configs/cascade_mask_rcnn_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_cascade_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/cascade_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'cascade_rcnn_r50_caffe_c4_1x-7c85c62b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_faster_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/faster_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'faster_rcnn_r50_caffe_c4_1x-75ecfdfa.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_mask_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/mask_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'mask_rcnn_r50_caffe_c4_1x-02a4ad3b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    def test_ssd300_coco(self):
        origin_config = 'configs/ssd300_coco.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'
        self.run_test(origin_config, self.download_if_not_yet(url))

    # Export

    def test_export_atss__atss_r50_fpn_1x(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'models/atss/atss_r50_fpn_1x_20200113-a7aa251e.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_dcn_faster_rcnn_dconv_c3_c5_r50_fpn_1x(self):
        origin_config = 'configs/dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'dcn/faster_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-e41688c9.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_gn__mask_rcnn_r50_fpn_gn_2x(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn_2x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'gn/mask_rcnn_r50_fpn_gn_2x_20180113-86832cf2.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_gn_ws__faster_rcnn_r50_fpn_gn_ws_1x(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'ws/faster_rcnn_r50_fpn_gn_ws_1x_20190418-935d00b6.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_fcos__fcos_r50_caffe_fpn_gn_1x_4gpu(self):
        origin_config = 'configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_20190516-9f253a93.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_foveabox__fovea_r50_fpn_4gpu_1x(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4gpu_1x.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/' \
              'foveabox/fovea_r50_fpn_4gpu_1x_20190905-3b185a5d.pth'
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
    #
    def test_export_cascade_mask_rcnn_r50_fpn_1x(self):
        origin_config = 'configs/cascade_mask_rcnn_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_cascade_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/cascade_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'cascade_rcnn_r50_caffe_c4_1x-7c85c62b.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_faster_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/faster_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'faster_rcnn_r50_caffe_c4_1x-75ecfdfa.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_mask_rcnn_r50_caffe_c4_1x(self):
        origin_config = 'configs/mask_rcnn_r50_caffe_c4_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'mask_rcnn_r50_caffe_c4_1x-02a4ad3b.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_export_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet_r50_fpn_1x.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))

    def test_export_ssd300_coco(self):
        origin_config = 'configs/ssd300_coco.py'
        url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/' \
              'ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth'
        self.run_export_test(origin_config, self.download_if_not_yet(url))


if __name__ == '__main__':
    unittest.main()
