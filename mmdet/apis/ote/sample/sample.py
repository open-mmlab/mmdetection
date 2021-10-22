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
import logging
import sys

import numpy as np
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from mmdet.apis.ote.apis.detection.ote_utils import get_task_class

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample showcasing the new API')
    parser.add_argument('template_file_path', help='path to template file')
    parser.add_argument('--export', action='store_true')
    return parser.parse_args()


def load_test_dataset():
    from ote_sdk.entities.annotation import Annotation, AnnotationSceneEntity, AnnotationSceneKind
    from ote_sdk.entities.dataset_item import DatasetItemEntity
    from ote_sdk.entities.image import Image
    from ote_sdk.entities.label import LabelEntity
    from ote_sdk.entities.scored_label import ScoredLabel
    from ote_sdk.entities.shapes.rectangle import Rectangle
    from ote_sdk.entities.subset import Subset

    def gen_image(resolution, x1, y1, x2, y2):
        w, h = resolution
        image = np.full([h, w, 3], fill_value=255, dtype=np.uint8)
        image[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w), :] = np.array([0, 128, 128], dtype=np.uint8)[None, None, :]
        return (image, Rectangle(x1=x1, y1=y1, x2=x2, y2=y2))

    images = [
        gen_image((640, 480), 0.0, 0.0, 0.5, 0.5),
        gen_image((640, 480), 0.5, 0.0, 1.0, 0.5),
        gen_image((640, 480), 0.0, 0.5, 0.5, 1.0),
        gen_image((640, 480), 0.5, 0.5, 1.0, 1.0),
    ]
    labels = [
        LabelEntity(name='rect', domain="detection", id=0)
    ]

    def get_image(i, subset):
        image, bbox = images[i]
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=AnnotationSceneEntity(
                annotations=[Annotation(bbox, labels=[ScoredLabel(label=labels[0])])],
                kind=AnnotationSceneKind.ANNOTATION
            ),
            subset=subset,
        )

    items = [
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(2, Subset.VALIDATION),
        get_image(3, Subset.VALIDATION),
        get_image(0, Subset.TESTING),
        get_image(1, Subset.TESTING),
        get_image(2, Subset.TESTING),
        get_image(3, Subset.TESTING),
    ]
    return DatasetEntity(items), labels


def main(args):
    logger.info('Initialize dataset')
    dataset, labels_list = load_test_dataset()
    labels_schema = LabelSchemaEntity.from_labels(labels_list)

    logger.info(f'Train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    logger.info(f'Validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    logger.info('Load model template')
    model_template = parse_model_template(args.template_file_path)

    logger.info('Set hyperparameters')
    params = create(model_template.hyper_parameters.data)
    params.learning_parameters.num_iters = 5
    params.learning_parameters.learning_rate_warmup_iters = 1
    params.learning_parameters.batch_size = 2

    logger.info('Setup environment')
    environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema, model_template=model_template)

    logger.info('Create base Task')
    task_impl_path = model_template.entrypoints.base
    task_cls = get_task_class(task_impl_path)
    task = task_cls(task_environment=environment)

    logger.info('Train model')
    output_model = ModelEntity(
        dataset,
        environment.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)
    task.train(dataset, output_model)

    logger.info('Get predictions on the validation set')
    validation_dataset = dataset.get_subset(Subset.VALIDATION)
    predicted_validation_dataset = task.infer(
        validation_dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(
        model=output_model,
        ground_truth_dataset=validation_dataset,
        prediction_dataset=predicted_validation_dataset,
    )
    logger.info('Estimate quality on validation set')
    task.evaluate(resultset)
    logger.info(str(resultset.performance))

    if args.export:
        logger.info('Export model')
        exported_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task.export(ExportType.OPENVINO, exported_model)

        logger.info('Create OpenVINO Task')
        environment.model = exported_model
        openvino_task_impl_path = model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        openvino_task = openvino_task_cls(environment)

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Estimate quality on validation set')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

        logger.info('Run POT optimization')
        optimized_model = ModelEntity(
            dataset,
            environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        openvino_task.optimize(
            OptimizationType.POT,
            dataset.get_subset(Subset.TRAINING),
            optimized_model,
            OptimizationParameters())

        logger.info('Get predictions on the validation set')
        predicted_validation_dataset = openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        resultset = ResultSetEntity(
            model=optimized_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=predicted_validation_dataset,
        )
        logger.info('Performance of optimized model:')
        openvino_task.evaluate(resultset)
        logger.info(str(resultset.performance))

if __name__ == '__main__':
    sys.exit(main(parse_args()) or 0)
