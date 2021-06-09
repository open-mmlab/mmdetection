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
import importlib
import sys
import yaml

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.label import Label
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.utils.project_factory import ProjectFactory

from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter
from mmdet.datasets import CocoDataset


logger = logger_factory.get_logger("Sample")


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('template_file_path', help='path to template file')
    args = parser.parse_args()
    return args

def load_template(path):
    with open(path) as f:
        template = yaml.full_load(f)
    return template

def get_label(x, all_labels):
    label_name = CocoDataset.CLASSES[x]
    return [label for label in all_labels if label.name == label_name][0]

def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def main(args):
    template = load_template(args.template_file_path)
    template['hyper_parameters']['params'].setdefault('algo_backend', {})['template'] = args.template_file_path
    task_impl_path = template['task']['impl']
    task_cls = get_task_class(task_impl_path)

    project = ProjectFactory().create_project_single_task(
        name='otedet-sample-project',
        description='otedet-sample-project',
        label_names=CocoDataset.CLASSES,
        task_name='otedet-task')

    dataset = MMDatasetAdapter(
        train_ann_file='data/coco/annotations/instances_val2017.json',
        train_data_root='data/coco/val2017/',
        val_ann_file='data/coco/annotations/instances_val2017.json',
        val_data_root='data/coco/val2017/',
        test_ann_file='data/coco/annotations/instances_val2017.json',
        test_data_root='data/coco/val2017/')
    dataset.get_subset(Subset.VALIDATION)
    dataset.set_project_labels(project.get_labels())

    print(f"train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
    print(f"validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

    environment = TaskEnvironment(project=project, task_node=project.tasks[-1])
    params = task_cls.get_configurable_parameters(environment)
    task_cls.apply_template_configurable_parameters(params, template)
    params.algo_backend.template.value = args.template_file_path
    environment.set_configurable_parameters(params)

    task = task_cls(task_environment=environment)

    # Tweak parameters.
    params = task.get_configurable_parameters(environment)
    params.learning_parameters.learning_rate.value = 1e-5
    params.learning_parameters.learning_rate_schedule.value = 'cyclic'
    # params.learning_parameters.learning_rate_warmup_iters.value = 0
    params.learning_parameters.batch_size.value = 32
    params.learning_parameters.num_epochs.value = 1
    params.postprocessing.result_based_confidence_threshold.value = False
    params.postprocessing.confidence_threshold.value = 0.025
    environment.set_configurable_parameters(params)
    task.update_configurable_parameters(environment)

    logger.info('Start model training... [ROUND 0]')
    model = task.train(dataset=dataset)
    logger.info('Model training finished [ROUND 0]')

    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.analyse(
        validation_dataset.with_empty_annotations(),
        AnalyseParameters(is_evaluation=True))
    resultset = ResultSet(
        model=model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )

    performance = task.compute_performance(resultset)
    resultset.performance = performance

    print(resultset.performance)

    task.optimize_loaded_model()


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args) or 0)
