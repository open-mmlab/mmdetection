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
import os
import os.path as osp
import sys
import random
from typing import Tuple

import cv2 as cv
from tqdm import tqdm

from mmdet.datasets import CocoDataset
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.annotation import Annotation, AnnotationKind
from sc_sdk.entities.datasets import Dataset, Subset, NullDataset
from sc_sdk.entities.id import ID
from sc_sdk.entities.image import Image
from sc_sdk.entities.label import ScoredLabel
from sc_sdk.entities.project import Project, NullProject
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.url import URL
from sc_sdk.logging import logger_factory
from sc_sdk.tests.test_helpers import generate_training_dataset_of_all_annotated_media_in_project
from sc_sdk.usecases.repos import *
from sc_sdk.utils.project_factory import ProjectFactory

from nousapi.apis.detection import MMObjectDetectionTask


logger = logger_factory.get_logger("Sample")


def create_project(projectname, taskname, classes):
    project = ProjectFactory().create_project_single_task(name=projectname, description="",
        label_names=classes, task_name=taskname)
    ProjectRepo().save(project)
    logger.info(f'New project created {str(project)}')
    return project

def load_project(projectname, taskname, classes):
    project = ProjectRepo().get_latest_by_name(projectname)
    if isinstance(project, NullProject):
        project = create_project(projectname, taskname, classes)
    else:
        logger.info(f'Existing project loaded {str(project)}')
    return project

def get_label(x, all_labels):
    label_name = CocoDataset.CLASSES[x]
    return [label for label in all_labels if label.name == label_name][0]

def create_coco_dataset(project, cfg=None):
    pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]
    coco_dataset = CocoDataset(ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/', pipeline=pipeline)

    logger.info(f'Loading images and annotation from {str(coco_dataset)} to repos')

    for datum in tqdm(coco_dataset):
        imdata = datum['img']
        imshape = imdata.shape
        image = Image(name=datum['ori_filename'], project=project, numpy=imdata)
        ImageRepo(project).save(image)

        gt_bboxes = datum['gt_bboxes']
        gt_labels = datum['gt_labels']

        shapes = []
        for label, bbox in zip(gt_labels, gt_bboxes):
            project_label = get_label(label, project.get_labels())
            shapes.append(
                Box(x1=float(bbox[0] / imshape[1]),
                    y1=float(bbox[1] / imshape[0]),
                    x2=float(bbox[2] / imshape[1]),
                    y2=float(bbox[3] / imshape[0]),
                    labels=[ScoredLabel(project_label)]))
        annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image.media_identifier, shapes=shapes)
        AnnotationRepo(project).save(annotation)

    dataset = generate_training_dataset_of_all_annotated_media_in_project(project)
    DatasetRepo(project).save(dataset)
    logger.info(f'New dataset created {dataset}')
    return dataset


def load_dataset(project, dataset_id=None):
    dataset = NullDataset()
    if dataset_id is not None:
        dataset = DatasetRepo(project).get_by_id(dataset_id)
    if isinstance(dataset, NullDataset):
        dataset = create_coco_dataset(project)
    else:
        logger.info(f'Existing dataset loaded {str(dataset)}')
    return dataset


projectname = "MMObjectDetectionSample"
project = load_project(projectname, "MMObjectDetectionTask", CocoDataset.CLASSES)
print('Tasks:', [task.task_name for task in project.tasks])

dataset = load_dataset(project, dataset_id=ID('60a395e50f3cb0a2e333e57e'))
print(dataset)
# dataset = create_coco_dataset(project)
print(f"train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
print(f"validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")

environment = TaskEnvironment(project=project, task_node=project.tasks[-1])
task = MMObjectDetectionTask(task_environment=environment)

# Tweak parameters.
params = task.get_configurable_parameters(environment)
available_models = params.learning_architecture.available_models
logger.info('Available models: \n\t' + '\n\t'.join(x['name'] for x in available_models))
params.learning_architecture.model_architecture.value = available_models[0]['name']
logger.warning(params.learning_architecture.model_architecture.value)
# params.learning_parameters.learning_rate.value = 1e-3
params.learning_parameters.learning_rate_schedule.value = 'cyclic'
# params.learning_parameters.learning_rate_warmup_iters.value = 0
params.learning_parameters.batch_size.value = 64
params.learning_parameters.num_epochs.value = 1
environment.set_configurable_parameters(params)
task.update_configurable_parameters(environment)

logger.info('Start model training... [ROUND 0]')
model = task.train(dataset=dataset)
ModelRepo(project).save(model)
logger.info('Model training finished [ROUND 0]')


# Tweak parameters.
params = task.get_configurable_parameters(environment)
available_models = params.learning_architecture.available_models
logger.info('Available models: \n\t' + '\n\t'.join(x['name'] for x in available_models))
params.learning_architecture.model_architecture.value = available_models[1]['name']
logger.warning(params.learning_architecture.model_architecture.value)
# params.learning_parameters.learning_rate.value = 1e-3
params.learning_parameters.learning_rate_schedule.value = 'cyclic'
# params.learning_parameters.learning_rate_warmup_iters.value = 0
params.learning_parameters.batch_size.value = 32
params.learning_parameters.num_epochs.value = 1
environment.set_configurable_parameters(params)
task.update_configurable_parameters(environment)

logger.info('Start model training... [ROUND 1]')
model = task.train(dataset=dataset)
ModelRepo(project).save(model)
logger.info('Model training finished [ROUND 1]')

validation_dataset = dataset.get_subset(Subset.VALIDATION)
predicted_validation_dataset = task.analyse(
    validation_dataset.with_empty_annotations(),
    AnalyseParameters(is_evaluation=True))
resultset = ResultSet(
    model=model,
    ground_truth_dataset=validation_dataset,
    prediction_dataset=predicted_validation_dataset,
)
ResultSetRepo(project).save(resultset)

performance = task.compute_performance(resultset)
resultset.performance = performance
ResultSetRepo(project).save(resultset)

print(resultset.performance)

task.optimize_loaded_model()
