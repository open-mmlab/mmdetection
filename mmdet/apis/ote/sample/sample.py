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

import argparse
import os.path as osp
import sys

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.dataset_storage import NullDatasetStorage
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.id import ID
from sc_sdk.entities.model import NullModel, Model, ModelStatus
from sc_sdk.entities.workspace import NullWorkspace
from sc_sdk.entities.model_template import NullModelTemplate
from sc_sdk.entities.model_storage import ModelStorage
from sc_sdk.entities.optimized_model import ModelOptimizationType, ModelPrecision, OptimizedModel, TargetDevice
from sc_sdk.entities.project import NullProject
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.tasks.interfaces.export_task import ExportType

from mmdet.apis.ote.apis.detection import MMObjectDetectionTask
from mmdet.apis.ote.apis.detection.configuration import ObjectDetectionConfig
from mmdet.apis.ote.apis.detection.config_utils import apply_template_configurable_parameters
from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter
from mmdet.apis.ote.apis.detection.ote_utils import generate_label_schema, load_template, get_task_class


logger = logger_factory.get_logger('Sample')


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    template = load_template(args.template_file_path)
    task_impl_path = template['task']['impl']
    task_cls = get_task_class(task_impl_path)

    dataset = MMDatasetAdapter(
        train_ann_file=osp.join(args.data_dir, 'coco/annotations/instances_val2017.json'),
        train_data_root=osp.join(args.data_dir, 'coco/val2017/'),
        val_ann_file=osp.join(args.data_dir, 'coco/annotations/instances_val2017.json'),
        val_data_root=osp.join(args.data_dir, 'coco/val2017/'),
        test_ann_file=osp.join(args.data_dir, 'coco/annotations/instances_val2017.json'),
        test_data_root=osp.join(args.data_dir, 'coco/val2017/'),
        dataset_storage=NullDatasetStorage)
    dataset.get_subset(Subset.VALIDATION)

    labels_schema = generate_label_schema(dataset.get_labels())
    labels_list = labels_schema.get_labels(False)
    dataset.set_project_labels(labels_list)

    print(f'train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    print(f'validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    params = ObjectDetectionConfig(workspace_id=ID(), project_id=ID(), task_id=ID())
    apply_template_configurable_parameters(params, template)
    environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)

    task: MMObjectDetectionTask = task_cls(task_environment=environment)

    output_model = Model(
            NullProject(),
            ModelStorage(NullWorkspace(), 'storage', NullModelTemplate()),
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

    # Tweak parameters.
    params = task.get_configurable_parameters(environment)
    params.learning_parameters.num_iters = 1000
    environment.set_configurable_parameters(params)
    task.update_configurable_parameters(environment)

    logger.info('Start model training... [ROUND 0]')
    task.train(dataset, output_model)
    logger.info('Model training finished [ROUND 0]')

    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.analyse(
        validation_dataset.with_empty_annotations(),
        AnalyseParameters(is_evaluation=True))
    resultset = ResultSet(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    performance = task.compute_performance(resultset)
    print(performance)

    if args.export:
        exported_model = OptimizedModel(
            NullProject(),
            ModelStorage(NullWorkspace(), 'storage', NullModelTemplate()),
            dataset,
            environment.get_model_configuration(),
            ModelOptimizationType.MO,
            [ModelPrecision.FP16],
            optimization_methods=[],
            optimization_level={},
            target_device=TargetDevice.UNSPECIFIED,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        task.export(ExportType.OPENVINO, exported_model)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
