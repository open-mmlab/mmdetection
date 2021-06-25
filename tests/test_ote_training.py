#import functools
#import numpy as np
#import os.path as osp
#import pytest
#import random
#import time
#import warnings
#from concurrent.futures import ThreadPoolExecutor
#
#from flaky import flaky
#from sc_sdk.entities.annotation import Annotation, AnnotationScene, AnnotationSceneKind
#from sc_sdk.entities.dataset_item import DatasetItem
#from sc_sdk.entities.datasets import Dataset, Subset
#from sc_sdk.entities.image import Image
#from sc_sdk.entities.media_identifier import ImageIdentifier
#from sc_sdk.entities.model import NullModel
#from sc_sdk.entities.optimized_model import OptimizedModel
#from sc_sdk.entities.resultset import ResultSet
#from sc_sdk.entities.shapes.box import Box
#from sc_sdk.entities.shapes.ellipse import Ellipse
#from sc_sdk.entities.shapes.polygon import Polygon
#from sc_sdk.entities.task_environment import TaskEnvironment
#from sc_sdk.tests.test_helpers import generate_random_annotated_image, rerun_on_flaky_assert
#from sc_sdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
#from sc_sdk.utils.project_factory import ProjectFactory
#
#from mmdet.apis.ote.apis.detection import MMObjectDetectionTask, MMDetectionParameters, configurable_parameters

#######
import importlib
import os.path as osp
import pytest
import sys
import yaml

from collections import namedtuple

from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Subset
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.utils.project_factory import ProjectFactory

from mmdet.apis.ote.extension.datasets.mmdataset import MMDatasetAdapter

from e2e_test_system import e2e_pytest


logger = logger_factory.get_logger('Sample')

def TRAINING_PARAMETERS_FIELDS():
    return ['annotations_train',
            'images_train_dir',
            'annotations_val',
            'images_val_dir',
            'annotations_test',
            'images_test_dir',
            'template_file_path',
            ]
TrainingParameters = namedtuple('TrainingParameters', TRAINING_PARAMETERS_FIELDS())

@pytest.fixture
def dataset_definitions_fx(request):
    path = request.config.getoption('--dataset-definitions')
    assert path is not None, (f'The command line parameter "--dataset-definitions" is not set, '
                             f'whereas it is required for the test {request.node.originalname or request.node.name}')
    with open(path) as f:
        data = yaml.safe_load(f)
    return data

def load_template(path):
    with open(path) as f:
        template = yaml.full_load(f)
    # Save path to template file, to resolve relative paths later.
    template['hyper_parameters']['params'].setdefault('algo_backend', {})['template'] = path
    return template

def get_task_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def run_ote_training(params: TrainingParameters):
    # check consistency
    print(f'Using for train annotation file {params.annotations_train}')
    print(f'Using for val annotation file {params.annotations_val}')
    train_ann_file = params.annotations_train
    train_data_root = params.images_train_dir
    val_ann_file = params.annotations_val
    val_data_root = params.images_val_dir
    test_ann_file = params.annotations_test
    test_data_root = params.images_test_dir

    dataset = MMDatasetAdapter(
        train_ann_file=train_ann_file,
        train_data_root=train_data_root,
        val_ann_file=val_ann_file,
        val_data_root=val_data_root,
        test_ann_file=test_ann_file,
        test_data_root=test_data_root)

    template = load_template(params.template_file_path)
    task_impl_path = template['task']['impl']
    task_cls = get_task_class(task_impl_path)


    project = ProjectFactory().create_project_single_task(
        name='otedet-sample-project',
        description='otedet-sample-project',
        label_names=dataset.get_labels(),
        task_name='otedet-task')

    dataset.set_project_labels(project.get_labels())

    print(f'train dataset: {len(dataset.get_subset(Subset.TRAINING))} items')
    print(f'validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items')

    environment = TaskEnvironment(project=project, task_node=project.tasks[-1])
    params = task_cls.get_configurable_parameters(environment)
    task_cls.apply_template_configurable_parameters(params, template)
    environment.set_configurable_parameters(params)

    task = task_cls(task_environment=environment)

    # Tweak parameters.
    params = task.get_configurable_parameters(environment)
    params.learning_parameters.nncf_quantization.value = False
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


def _get_training_params_from_dataset_definitions(dataset_definitions, dataset_name):
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items()
                                  if k in TRAINING_PARAMETERS_FIELDS()}

    assert set(TRAINING_PARAMETERS_FIELDS()) == set(training_parameters_fields.keys()), \
            f'ERROR: dataset definitions for name={dataset_name} does not contain all required fields'
    assert all(training_parameters_fields.values()), \
            f'ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields'

    params = TrainingParameters(**training_parameters_fields)
    return params

@e2e_pytest
@pytest.mark.parametrize('dataset_name',
                         ['coco_shortened_500',
                          'vitens_tiled_shortened_500'])
def test_ote_training(dataset_name, dataset_definitions_fx):
    training_params = _get_training_params_from_dataset_definitions(dataset_definitions_fx, dataset_name)
    run_ote_training(training_params)
