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
import os.path as osp
import sys
import unittest
from shutil import copy2 as copy
from subprocess import run, CalledProcessError, PIPE

from mmcv import Config

from common import replace_text_in_file, collect_ap


class PublicModelsTestCase(unittest.TestCase):
    root_dir = '/tmp'
    coco_dir = osp.join(root_dir, 'data/coco')
    snapshots_dir = osp.join(root_dir, 'snapshots')

    custom_operations = ['ExperimentalDetectronROIFeatureExtractor',
                         'PriorBox', 'PriorBoxClustered', 'DetectionOutput',
                         'DeformableConv2D']

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted([item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}', check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
            check=True, shell=True)

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 10

        cls.annotation_file = osp.join(cls.coco_dir, f'annotations/instances_val2017_short_{cls.shorten_to}.json')
        cls.shorten_annotation(osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               cls.annotation_file, cls.shorten_to)

    def prerun(self, config_path, test_dir):
        log_file = osp.join(test_dir, 'test.log')
        os.makedirs(test_dir, exist_ok=True)
        target_config_path = osp.join(test_dir, 'config.py')
        cfg = Config.fromfile(config_path)
        update_args = {
            'data_root': f'{self.coco_dir}/',
            'data.test.ann_file': self.annotation_file,
            'data.test.img_prefix': osp.join(self.coco_dir, 'val2017/'),
        }
        cfg.merge_from_dict(update_args)
        with open(target_config_path, 'wt') as config_file:
            config_file.write(cfg.pretty_text)
        return log_file, target_config_path

    def postrun(self, log_file, expected_output_file, metrics, thr):
        print('expected ouputs', expected_output_file)
        ap = collect_ap(log_file)
        with open(expected_output_file) as read_file:
            content = json.load(read_file)
        reference_ap = content['map']
        print(f'expected {reference_ap} vs actual {ap}')
        for expected, actual, m in zip(reference_ap, ap, metrics):
            if abs(actual - expected) > thr:
                raise AssertionError(f'{m}: {expected} (expected) - {thr} (threshold) > {actual}')

    def domain_check_for_custom_operations(self, config_dir):
        config_onnx = osp.join(config_dir, 'model.onnx')
        from onnx import load
        onnx_model = load(config_onnx)

        from mmdet.utils.deployment.operations_domain import DOMAIN_CUSTOM_OPS_NAME
        for op_node in onnx_model.graph.node:
            if op_node.op_type in self.custom_operations:
                if op_node.domain != DOMAIN_CUSTOM_OPS_NAME:
                    error = f'In model {config_dir}, custom operation "{op_node.op_type}" does not have domain {DOMAIN_CUSTOM_OPS_NAME}.'
                    raise ValueError(error)

    def run_pytorch_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.0):
        print('\n\ntesting ' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'pytorch')
        log_file, target_config_path = self.prerun(config_path, test_dir)

        metrics_str = ' '.join(metrics)

        with open(log_file, 'w') as log_f:
            error = None
            try:
                run(f'python tools/test.py '
                    f'{target_config_path} '
                    f'{snapshot} '
                    f'--out {test_dir}/res.pkl --eval {metrics_str}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

        expected_output_file = f'tests/expected_outputs/public/{name}-{self.shorten_to}.json'
        self.postrun(log_file, expected_output_file, metrics, thr)

    def run_openvino_export_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.02, alt_ssd_export=False):
        print('\n\ntesting OpenVINO export ' + '(--alt_ssd_export)' if alt_ssd_export else '' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'openvino_alt_ssd_export' if alt_ssd_export else 'openvino_export')
        log_file, target_config_path = self.prerun(config_path, test_dir)

        metrics_str = ' '.join(metrics)

        with open(log_file, 'w') as log_f:
            error = None
            try:
                run(f'python tools/export.py '
                    f'{target_config_path} '
                    f'{snapshot} '
                    f'{test_dir} '
                    f'openvino {"--alt_ssd_export" if alt_ssd_export else ""}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Export script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

            self.domain_check_for_custom_operations(test_dir)

            try:
                run(f'python tools/test_exported.py '
                    f'{target_config_path} '
                    f'{osp.join(test_dir, "model.xml")} '
                    f'--out res.pkl --eval {metrics_str} 2>&1 | tee {log_file}',
                    stdout=log_f, stderr=PIPE, check=True, shell=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

        expected_output_file = f'tests/expected_outputs/public/{name}-{self.shorten_to}.json'
        self.postrun(log_file, expected_output_file, metrics, thr)

    def run_onnx_export_test(self, config_path, snapshot, metrics=('bbox', ), thr=0.02):
        print('\n\ntesting ONNX export ' + config_path, file=sys.stderr)
        name = config_path.replace('configs/', '')[:-3]
        test_dir = osp.join(self.root_dir, name, 'onnx_export')
        log_file, target_config_path = self.prerun(config_path, test_dir)

        with open(log_file, 'w') as log_f:
            error = None
            try:
                run(['python', 'tools/export.py',
                    target_config_path,
                    snapshot,
                    test_dir,
                    'onnx'
                    ], stdout=log_f, stderr=PIPE, check=True)
            except CalledProcessError as ex:
                error = 'Export script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)

            try:
                run(['python',
                    'tools/test_exported.py',
                    target_config_path,
                    osp.join(test_dir, 'model.onnx'),
                    '--out', 'res.pkl',
                    '--eval', *metrics
                    ], stdout=log_f, stderr=PIPE, check=True)
            except CalledProcessError as ex:
                error = 'Test script failure.\n' + ex.stderr.decode(sys.getfilesystemencoding())
            if error is not None:
                raise RuntimeError(error)


        expected_output_file = f'tests/expected_outputs/public/{name}-{self.shorten_to}.json'
        self.postrun(log_file, expected_output_file, metrics, thr)

    def download_if_not_yet(self, url):
        os.makedirs(self.snapshots_dir, exist_ok=True)
        path = osp.join(self.snapshots_dir, osp.basename(url))
        if not osp.exists(path):
            run(f'wget {url} -P {self.snapshots_dir}', check=True, shell=True)
        return path

    def test_pytorch_atss__atss_r50_fpn_1x_coco(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_dcn__faster_rcnn_r50_fpn_dconv_c3_5_1x_coco(self):
        origin_config = 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_gn__mask_rcnn_r50_fpn_gn_all_2x_coco(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_pytorch_gn_ws__faster_rcnn_r50_fpn_gn_ws_all_1x_coco(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_fcos__fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_4x2_2x_coco(self):
        origin_config = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
              'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_foveabox__fovea_r50_fpn_4x4_1x_coco(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_ms_rcnn__ms_rcnn_r50_caffe_fpn_2x_coco(self):
        origin_config = 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    # def test_pytorch_htc__htc_r50_fpn_1x(self):
    #     origin_config = 'configs/htc/htc_r50_fpn_1x.py'
    #     url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/' \
    #           'htc_r50_fpn_1x_20190408-878c1712.pth'
    #     self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_cascade_rcnn__cascade_mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_pytorch_cascade_rcnn__cascade_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_faster_rcnn__faster_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_mask_rcnn__mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_pytorch_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_ssd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    def test_pytorch_retinanet_effd0_bifpn_1x_coco(self):
        origin_config = 'configs/efficientdet/retinanet_effd0_bifpn_1x_coco.py'
        url = 'https://storage.openvinotoolkit.org/repositories/mmdetection/models/efficientdet/' \
              'retinanet_effd0_bifpn_1x_coco/epoch_300.pth'
        self.run_pytorch_test(origin_config, self.download_if_not_yet(url))

    # Export

    def test_openvino_atss__atss_r50_fpn_1x_coco(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_dcn__faster_rcnn_r50_fpn_dconv_c3_5_1x_coco(self):
        origin_config = 'configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'
        url = 'https://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/' \
              'faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_gn__mask_rcnn_r50_fpn_gn_all_2x_coco(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_openvino_gn_ws__faster_rcnn_r50_fpn_gn_ws_all_1x_coco(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_fcos__fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_4x2_2x_coco(self):
        origin_config = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
              'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_foveabox__fovea_r50_fpn_4x4_1x_coco(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    # def test_openvino_ms_rcnn__ms_rcnn_r50_caffe_fpn_1x(self):
    #     origin_config = 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
    #           'ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth'
    #     self.run_openvino_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))
    #
    # def test_openvino_htc__htc_r50_fpn_1x(self):
    #     origin_config = 'configs/htc/htc_r50_fpn_1x.py'
    #     url = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/' \
    #           'htc_r50_fpn_1x_20190408-878c1712.pth'
    #     self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_cascade_rcnn__cascade_mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_openvino_cascade_rcnn__cascade_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_faster_rcnn__faster_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_mask_rcnn__mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_openvino_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_ssd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_openvino_alt_ssd_ssd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url), alt_ssd_export=True)

    def test_openvino_retinanet_effd0_bifpn_1x_coco(self):
        origin_config = 'configs/efficientdet/retinanet_effd0_bifpn_1x_coco.py'
        url = 'https://storage.openvinotoolkit.org/repositories/mmdetection/models/efficientdet/' \
              'retinanet_effd0_bifpn_1x_coco/epoch_300.pth'
        self.run_openvino_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_atss__atss_r50_fpn_1x_coco(self):
        origin_config = 'configs/atss/atss_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/' \
              'v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_gn__mask_rcnn_r50_fpn_gn_all_2x_coco(self):
        origin_config = 'configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_onnx_gn_ws__faster_rcnn_r50_fpn_gn_ws_all_1x_coco(self):
        origin_config = 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_fcos__fcos_x101_64x4d_fpn_gn_head_mstrain_640_800_4x2_2x_coco(self):
        origin_config = 'configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco/' \
              'fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco_20200229-11f8c079.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_foveabox__fovea_r50_fpn_4x4_1x_coco(self):
        origin_config = 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    # def test_onnx_ms_rcnn__ms_rcnn_r50_caffe_fpn_1x(self):
    #     origin_config = 'configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
    #           'ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth'
    #     self.run_onnx_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    # def test_onnx_htc__htc_r50_fpn_1x(self):
    #     origin_config = 'configs/htc/htc_r50_fpn_20e_coco.py'
    #     url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/htc/' \
    #           'htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth'
    #     self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_cascade_rcnn__cascade_mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_onnx_cascade_rcnn__cascade_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_faster_rcnn__faster_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_mask_rcnn__mask_rcnn_r50_fpn_1x_coco(self):
        origin_config = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url), ('bbox', 'segm'))

    def test_onnx_retinanet_r50_fpn_1x(self):
        origin_config = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_ssd300_coco(self):
        origin_config = 'configs/ssd/ssd300_coco.py'
        url = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/' \
              'ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))

    def test_onnx_retinanet_effd0_bifpn_1x_coco(self):
        origin_config = 'configs/efficientdet/retinanet_effd0_bifpn_1x_coco.py'
        url = 'https://storage.openvinotoolkit.org/repositories/mmdetection/models/efficientdet/' \
              'retinanet_effd0_bifpn_1x_coco/epoch_300.pth'
        self.run_onnx_export_test(origin_config, self.download_if_not_yet(url))


if __name__ == '__main__':
    unittest.main()
