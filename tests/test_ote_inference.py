# Copyright (C) 2021 Intel Corporation
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

import itertools
import json
import mmcv
import os
import os.path as osp
import torch
import unittest
from e2e_test_system import e2e_pytest_api
from mmcv.parallel import scatter
from subprocess import run

from mmdet.apis import init_detector
from mmdet.datasets import build_dataloader, build_dataset

MODEL_CONFIGS = [
    'configs/ote/custom-object-detection/gen3_resnet50_VFNet/model.py',
    'configs/ote/custom-object-detection/gen3_mobilenetV2_ATSS/model.py',
    'configs/ote/custom-object-detection/gen3_mobilenetV2_SSD/model.py'
]

DEVICES = ['cuda:0', 'cpu']


class TestInference(unittest.TestCase):
    root_dir = '/tmp'
    coco_dir = osp.join(root_dir, 'data/coco')

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted(
                [item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [
                item for item in content['images']
                if item['id'] in selected_indexes
            ]
            content['annotations'] = [
                item for item in content['annotations']
                if item['image_id'] in selected_indexes
            ]
            content['licenses'] = [
                item for item in content['licenses']
                if item['id'] in selected_indexes
            ]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
                check=True,
                shell=True)

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 10

        cls.src_anno = osp.join(cls.coco_dir,
                                'annotations/instances_val2017.json')
        cls.dst_anno = osp.join(
            cls.coco_dir,
            f'annotations/instances_val2017_short_to_{cls.shorten_to}.json')
        cls.shorten_annotation(cls.src_anno, cls.dst_anno, cls.shorten_to)

    @e2e_pytest_api
    def test_inference(self):
        for cfg, device in itertools.product(MODEL_CONFIGS, DEVICES):
            print(f'Starting inference test: {cfg} on {device}')
            self.run_test(cfg, device)

    def run_test(self, cfg_path, device):
        config = mmcv.Config.fromfile(cfg_path)
        config.data.test.ann_file = self.dst_anno
        config.data.test.img_prefix = osp.join(self.coco_dir, 'val2017')
        model = init_detector(config, config.load_from, device=device)
        dataset = build_dataset(config.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=config.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        results = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = [data['img'][0].data]
            if next(model.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [device])[0]
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            batch_size = len(result)
            results.extend(result)
            for _ in range(batch_size):
                prog_bar.update()
        dataset.evaluate(results)
