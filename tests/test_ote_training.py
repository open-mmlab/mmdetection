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

import glob
import itertools
import logging
import os
import os.path as osp
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
from copy import deepcopy
from pprint import pformat
from typing import Optional, Union

import pytest
import yaml
from e2e_test_system import DataCollector, e2e_pytest_performance
from ote_sdk.configuration.helper import create
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelPrecision,
    ModelStatus,
    ModelOptimizationType,
    OptimizationMethod,
)
from ote_sdk.entities.model_template import parse_model_template, TargetDevice
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import get_task_class
from mmdet.apis.ote.extension.datasets.data_utils import load_dataset_items_coco_format
from mmdet.integration.nncf.utils import is_nncf_enabled

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) #TODO(lbeynens): REMOVE BEFORE MERGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# This string constant will be used as a special constant for a config field
# value to point that the field should be filled in tests' code by some default
# value specific for this field.
def DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST():
    return 'DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST'

# This string constant will be used as a special constant for a config field value to point
# that the field should NOT be changed in tests -- its value should be taken
# from the template file or the config file of the model.
def KEEP_CONFIG_FIELD_VALUE():
    return 'KEEP_CONFIG_FIELD_VALUE'

def ABSENT_EXPECTED_VALIDATION_METRICS():
    return 'ABSENT_EXPECTED_VALIDATION_METRICS'

def SKIP_VALIDATION():
    return 'SKIP_VALIDATION'

def DATASET_PARAMETERS_FIELDS():
    return ('annotations_train',
            'images_train_dir',
            'annotations_val',
            'images_val_dir',
            'annotations_test',
            'images_test_dir',
            )

ROOT_PATH_KEY = '_root_path'
DatasetParameters = namedtuple('DatasetParameters', DATASET_PARAMETERS_FIELDS())

@pytest.fixture
def dataset_definitions_fx(request):
    """
    Return dataset definitions read from a YAML file passed as the parameter --dataset-definitions.
    Note that the dataset definitions should store the following structure:
    {
        <dataset_name>: {
            'annotations_train': <annotation_file_path1>
            'images_train_dir': <images_folder_path1>
            'annotations_val': <annotation_file_path2>
            'images_val_dir': <images_folder_path2>
            'annotations_test': <annotation_file_path3>
            'images_test_dir':  <images_folder_path3>
        }
    }
    """
    path = request.config.getoption('--dataset-definitions')
    if path is None:
        logger.warning(f'The command line parameter "--dataset-definitions" is not set'
                       f'whereas it is required for the test {request.node.originalname or request.node.name}'
                       f' -- ALL THE TESTS THAT REQUIRE THIS PARAMETER ARE SKIPPED')
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    data[ROOT_PATH_KEY] = osp.dirname(path)
    return data

@pytest.fixture
def template_paths_fx(request):
    """
    Return mapping model names to template paths, received from globbing the folder configs/ote/
    Note that the function searches files with name `template.yaml`, and for each such file
    the model name is the name of the parent folder of the file.
    """
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    glb = glob.glob(f'{root}/configs/ote/**/template.yaml', recursive=True)
    data = {}
    for p in glb:
        assert osp.isabs(p), f'Error: not absolute path {p}'
        name = osp.basename(osp.dirname(p))
        if name in data:
            raise RuntimeError(f'Duplication of names in config/ote/ folder: {data[name]} and {p}')
        data[name] = p
    data[ROOT_PATH_KEY] = ''
    return data

@pytest.fixture
def expected_metrics_all_tests_fx(request):
    """
    Return dataset definitions read from a YAML file passed as the parameter --dataset-definitions.
    Note that the dataset definitions should store the following structure:
    {
        ...
    }
    """
    path = request.config.getoption('--expected-metrics-file')
    if path is None:
        logger.warning(f'The command line parameter "--expected-metrics-file" is not set'
                       f'whereas it is required to compare with target metrics'
                       f' -- ALL THE COMPARISON WITH TARGET METRICS IN TESTS WILL BE TURNED OFF')
        return None
    with open(path) as f:
        expected_metrics_all_tests = yaml.safe_load(f)
    assert isinstance(expected_metrics_all_tests, dict), f'Wrong metrics file {path}: {expected_metrics_all_tests}'
    return expected_metrics_all_tests

@pytest.fixture
def current_test_parameters_fx(request):
    cur_test_params = deepcopy(request.node.callspec.params)
    assert 'test_parameters' in cur_test_params, \
            f'The test {request.node.name} should be parametrized by parameter "test_parameters"'
    return cur_test_params['test_parameters']

@pytest.fixture
def current_test_parameters_string_fx(request):
    node_name = request.node.name
    assert '[' in node_name, f'Wrong format of node name {node_name}'
    assert node_name.endswith(']'), f'Wrong format of node name {node_name}'
    index = node_name.find('[')
    return node_name[index+1:-1]

@pytest.fixture
def current_test_expected_metrics_fx(expected_metrics_all_tests_fx, current_test_parameters_string_fx,
                                     current_test_parameters_fx):
    if 'reallife' != current_test_parameters_fx['usecase']:
        return SKIP_VALIDATION()
    if expected_metrics_all_tests_fx is None:
        logger.error(f'The dict with expected metrics cannot be read -- most probably it will cause test fail')
        return ABSENT_EXPECTED_VALIDATION_METRICS()
    if current_test_parameters_string_fx not in expected_metrics_all_tests_fx:
        logger.error(f'The parameters id string {current_test_parameters_string_fx} is not inside '
                     f'the dict with expected metrics -- most probably it will cause test fail')
        return ABSENT_EXPECTED_VALIDATION_METRICS()
    return expected_metrics_all_tests_fx[current_test_parameters_string_fx]

def _make_path_be_abs(some_val, root_path):
    assert isinstance(some_val, (str, dict)), f'Wrong type of value: {some_val}, type={type(some_val)}'
    assert isinstance(root_path, str), f'Wrong type of root_path: {root_path}, type={type(root_path)}'

    # Note that os.path.join(a, b) == b if b is an absolute path
    if isinstance(some_val, str):
        return osp.join(root_path, some_val)

    some_dict = some_val
    assert all(isinstance(v, str) for v in some_dict.values()), f'Wrong input dict {some_dict}'
    for k in list(some_dict.keys()):
        some_dict[k] = osp.join(root_path, some_dict[k])
    return some_dict

def _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name):
    cur_dataset_definition = dataset_definitions[dataset_name]
    training_parameters_fields = {k: v for k, v in cur_dataset_definition.items()
                                  if k in DATASET_PARAMETERS_FIELDS()}
    _make_path_be_abs(training_parameters_fields, dataset_definitions[ROOT_PATH_KEY])

    assert set(DATASET_PARAMETERS_FIELDS()) == set(training_parameters_fields.keys()), \
            f'ERROR: dataset definitions for name={dataset_name} does not contain all required fields'
    assert all(training_parameters_fields.values()), \
            f'ERROR: dataset definitions for name={dataset_name} contains empty values for some required fields'

    params = DatasetParameters(**training_parameters_fields)
    return params

def performance_to_score_name_value(perf: Union[Performance, None]):
    """
    The method is intended to get main score info from Performance class
    """
    if perf is None:
        return None, None
    assert isinstance(perf, Performance)
    score = perf.score
    assert isinstance(score, ScoreMetric)
    assert isinstance(score.name, str) and score.name, f'Wrong score name "{score.name}"'
    return score.name, score.value

def convert_hyperparams_to_dict(hyperparams):
    def _convert(p):
        if p is None:
            return None
        d = {}
        groups = getattr(p, 'groups', [])
        parameters = getattr(p, 'parameters', [])
        assert (not groups) or isinstance(groups, list), f'Wrong field "groups" of p={p}'
        assert (not parameters) or isinstance(parameters, list), f'Wrong field "parameters" of p={p}'
        for group_name in groups:
            g = getattr(p, group_name, None)
            d[group_name] = _convert(g)
        for par_name in parameters:
            d[par_name] = getattr(p, par_name, None)
        return d
    return _convert(hyperparams)

class BaseOTETestAction(ABC):
    _name = None
    _with_validation = False

    @property
    def name(self):
        return type(self)._name

    @property
    def with_validation(self):
        return type(self)._with_validation

    def _check_result_prev_stages(self, results_prev_stages, list_required_stages):
        for stage_name in list_required_stages:
            if not results_prev_stages or stage_name not in results_prev_stages:
                raise RuntimeError(f'The action {self.name} requires results of the stage {stage_name}, '
                                   f'but they are absent')

    @abstractmethod
    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        raise NotImplementedError('The main action method is not implemented')

class OTETestTrainingAction(BaseOTETestAction):
    _name = 'training'
    def __init__(self, dataset_params, template_file_path, num_training_iters, batch_size):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)
        logger.info('Create base Task')
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, 'performance', None)
        if training_performance is None:
            raise RuntimeError('Cannot get training performance')
        return performance_to_score_name_value(training_performance)

    def _run_ote_training(self, data_collector):
        logger.debug(f'self.template_file_path = {self.template_file_path}')
        logger.debug(f'Using for train annotation file {self.dataset_params.annotations_train}')
        logger.debug(f'Using for val annotation file {self.dataset_params.annotations_val}')

        labels_list = []
        items = load_dataset_items_coco_format(
            ann_file_path=self.dataset_params.annotations_train,
            data_root_dir=self.dataset_params.images_train_dir,
            subset=Subset.TRAINING,
            labels_list=labels_list)
        items.extend(load_dataset_items_coco_format(
            ann_file_path=self.dataset_params.annotations_val,
            data_root_dir=self.dataset_params.images_val_dir,
            subset=Subset.VALIDATION,
            labels_list=labels_list))
        items.extend(load_dataset_items_coco_format(
            ann_file_path=self.dataset_params.annotations_test,
            data_root_dir=self.dataset_params.images_test_dir,
            subset=Subset.TESTING,
            labels_list=labels_list))
        self.dataset = DatasetEntity(items=items)

        self.labels_schema = LabelSchemaEntity.from_labels(labels_list)

        print(f'train dataset: {len(self.dataset.get_subset(Subset.TRAINING))} items')
        print(f'validation dataset: {len(self.dataset.get_subset(Subset.VALIDATION))} items')

        logger.debug('Load model template')
        self.model_template = parse_model_template(self.template_file_path)

        hyper_parameters = self.model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)

        logger.debug('Setup environment')
        params = create(hyper_parameters)
        logger.debug('Set hyperparameters')
        if self.num_training_iters != KEEP_CONFIG_FIELD_VALUE():
            params.learning_parameters.num_iters = int(self.num_training_iters)
            logger.debug(f'Set params.learning_parameters.num_iters={params.learning_parameters.num_iters}')
        else:
            logger.debug(f'Keep params.learning_parameters.num_iters={params.learning_parameters.num_iters}')

        if params.learning_parameters.num_iters < 20:
            num_checkpoints = 2
        elif params.learning_parameters.num_iters < 1000:
            num_checkpoints = 10
        else:
            num_checkpoints = 30

        if self.batch_size != KEEP_CONFIG_FIELD_VALUE():
            params.learning_parameters.batch_size = int(self.batch_size)
            logger.debug(f'Set params.learning_parameters.batch_size={params.learning_parameters.batch_size}')
        else:
            logger.debug(f'Keep params.learning_parameters.batch_size={params.learning_parameters.batch_size}')

        params.learning_parameters.num_checkpoints = num_checkpoints

        self.environment, self.task = self._create_environment_and_task(params,
                                                                        self.labels_schema,
                                                                        self.model_template)

        logger.debug('Train model')
        self.output_model = ModelEntity(
            self.dataset,
            self.environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

        self.copy_hyperparams = deepcopy(self.task._hyperparams)

        self.task.train(self.dataset, self.output_model)
        assert self.output_model.model_status == ModelStatus.SUCCESS, 'Training was failed'

        score_name, score_value = self._get_training_performance_as_score_name_value()
        logger.info(f'performance={self.output_model.performance}')
        data_collector.log_final_metric('training_accuracy/' + score_name, score_value)

        hyperparams_dict = convert_hyperparams_to_dict(self.copy_hyperparams)
        for k, v in hyperparams_dict.items():
            data_collector.update_metadata(k, v)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._run_ote_training(data_collector)
        results = {
                'model_template': self.model_template,
                'task': self.task,
                'dataset': self.dataset,
                'environment': self.environment,
                'output_model': self.output_model,
        }
        return results

def run_evaluation(dataset, task, model):
    logger.debug('Evaluation: Get predictions on the dataset')
    predicted_dataset = task.infer(
        dataset.with_empty_annotations(),
        InferenceParameters(is_evaluation=True))
    resultset = ResultSetEntity(
        model=model,
        ground_truth_dataset=dataset,
        prediction_dataset=predicted_dataset,
    )
    logger.debug('Evaluation: Estimate quality on dataset')
    task.evaluate(resultset)
    evaluation_performance = resultset.performance
    logger.info(f'Evaluation: performance={evaluation_performance}')
    score_name, score_value = performance_to_score_name_value(evaluation_performance)
    return score_name, score_value

class OTETestTrainingEvaluationAction(BaseOTETestAction):
    _name = 'training_evaluation'
    _with_validation = True

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_evaluation(self, data_collector,
                            dataset, task, trained_model):
        logger.info('Begin evaluation of trained model')
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(validation_dataset, task, trained_model)
        data_collector.log_final_metric('evaluation_accuracy/' + score_name, score_value)
        logger.info(f'End evaluation of trained model, results: {score_name}: {score_value}')
        return score_name, score_value

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training'])

        kwargs = {
                'dataset': results_prev_stages['training']['dataset'],
                'task': results_prev_stages['training']['task'],
                'trained_model': results_prev_stages['training']['output_model'],
        }

        score_name, score_value = self._run_ote_evaluation(data_collector, **kwargs)
        results = {
                'metrics': {
                    'accuracy': {
                        score_name: score_value
                    }
                }
        }
        return results

def run_export(environment, dataset, task, action_name):
    logger.debug(f'For action "{action_name}": Copy environment for evaluation exported model')

    environment_for_export = deepcopy(environment)

    logger.debug(f'For action "{action_name}": Create exported model')
    exported_model = ModelEntity(
        dataset,
        environment_for_export.get_model_configuration(),
        model_status=ModelStatus.NOT_READY)
    logger.debug('Run export')
    task.export(ExportType.OPENVINO, exported_model)

    assert exported_model.model_status == ModelStatus.SUCCESS, \
            f'In action "{action_name}": Export to OpenVINO was not successful'
    assert exported_model.model_format == ModelFormat.OPENVINO, \
            f'In action "{action_name}": Wrong model format after export'
    assert exported_model.optimization_type == ModelOptimizationType.MO, \
            f'In action "{action_name}": Wrong optimization type'

    logger.debug(f'For action "{action_name}": Set exported model into environment for export')
    environment_for_export.model = exported_model
    return environment_for_export, exported_model

class OTETestExportAction(BaseOTETestAction):
    _name = 'export'

    def _run_ote_export(self, data_collector,
                        environment, dataset, task):
        self.environment_for_export, self.exported_model = \
                run_export(environment, dataset, task, action_name=self.name)

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training'])

        kwargs = {
                'environment': results_prev_stages['training']['environment'],
                'dataset': results_prev_stages['training']['dataset'],
                'task': results_prev_stages['training']['task'],
        }

        self._run_ote_export(data_collector, **kwargs)
        results = {
                'environment': self.environment_for_export,
                'exported_model': self.exported_model,
        }
        return results

def create_openvino_task(model_template, environment):
    logger.debug('Create OpenVINO Task')
    openvino_task_impl_path = model_template.entrypoints.openvino
    openvino_task_cls = get_task_class(openvino_task_impl_path)
    openvino_task = openvino_task_cls(environment)
    return openvino_task

class OTETestExportEvaluationAction(BaseOTETestAction):
    _name = 'export_evaluation'
    _with_validation = True

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_export_evaluation(self, data_collector,
                                   model_template, dataset,
                                   environment_for_export, exported_model):
        logger.info('Begin evaluation of exported model')
        self.openvino_task = create_openvino_task(model_template, environment_for_export)
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(validation_dataset, self.openvino_task, exported_model)
        data_collector.log_final_metric('evaluation_accuracy_exported/' + score_name, score_value)
        logger.info('End evaluation of exported model')
        return score_name, score_value

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'export'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'environment_for_export': results_prev_stages['export']['environment'],
                'exported_model': results_prev_stages['export']['exported_model'],
        }

        score_name, score_value = self._run_ote_export_evaluation(data_collector, **kwargs)
        results = {
                'metrics': {
                    'accuracy': {
                        score_name: score_value
                    }
                }
        }
        return results

class OTETestPotAction(BaseOTETestAction):
    _name = 'pot'

    def __init__(self, pot_subset=Subset.TRAINING):
        self.pot_subset = pot_subset

    def _run_ote_pot(self, data_collector,
                     model_template, dataset,
                     environment_for_export):
        logger.debug('Creating environment and task for POT optimization')
        self.environment_for_pot = deepcopy(environment_for_export)
        self.openvino_task_pot = create_openvino_task(model_template, environment_for_export)

        self.optimized_model_pot = ModelEntity(
            dataset,
            self.environment_for_pot.get_model_configuration(),
            optimization_type=ModelOptimizationType.POT,
            optimization_methods=OptimizationMethod.QUANTIZATION,
            optimization_objectives={},
            precision=[ModelPrecision.INT8],
            target_device=TargetDevice.CPU,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        logger.info('Run POT optimization')
        self.openvino_task_pot.optimize(
            OptimizationType.POT,
            dataset.get_subset(self.pot_subset),
            self.optimized_model_pot,
            OptimizationParameters())
        assert self.optimized_model_pot.model_status == ModelStatus.SUCCESS, 'POT optimization was not successful'
        assert self.optimized_model_pot.model_format == ModelFormat.OPENVINO, 'Wrong model format after pot'
        assert self.optimized_model_pot.optimization_type == ModelOptimizationType.POT, 'Wrong optimization type'
        logger.info('POT optimization is finished')

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['export'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'environment_for_export': results_prev_stages['export']['environment'],
        }

        self._run_ote_pot(data_collector, **kwargs)
        results = {
                'openvino_task_pot': self.openvino_task_pot,
                'optimized_model_pot': self.optimized_model_pot,
        }
        return results

class OTETestPotEvaluationAction(BaseOTETestAction):
    _name = 'pot_evaluation'
    _with_validation = True

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_pot_evaluation(self, data_collector,
                                dataset,
                                openvino_task_pot,
                                optimized_model_pot):
        logger.info('Begin evaluation of pot model')
        validation_dataset_pot = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(validation_dataset_pot, openvino_task_pot, optimized_model_pot)
        data_collector.log_final_metric('evaluation_accuracy_pot/' + score_name, score_value)
        logger.info('End evaluation of pot model')
        return score_name, score_value

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'pot'])

        kwargs = {
                'dataset': results_prev_stages['training']['dataset'],
                'openvino_task_pot': results_prev_stages['pot']['openvino_task_pot'],
                'optimized_model_pot': results_prev_stages['pot']['optimized_model_pot'],
        }

        score_name, score_value = self._run_ote_pot_evaluation(data_collector, **kwargs)
        results = {
                'metrics': {
                    'accuracy': {
                        score_name: score_value
                    }
                }
        }
        return results

class OTETestNNCFAction(BaseOTETestAction):
    _name = 'nncf'

    def _run_ote_nncf(self, data_collector,
                      model_template, dataset, trained_model,
                      environment):
        logger.debug('Get predictions on the validation set for exported model')
        self.environment_for_nncf = deepcopy(environment)

        logger.info('Create NNCF Task')
        nncf_task_class_impl_path = model_template.entrypoints.nncf
        if not nncf_task_class_impl_path:
            pytest.skip('NNCF is not enabled for this template')

        if not is_nncf_enabled():
            pytest.skip('NNCF is not installed')

        logger.info('Creating NNCF task and structures')
        self.nncf_model = ModelEntity(
            dataset,
            self.environment_for_nncf.get_model_configuration(),
            optimization_type=ModelOptimizationType.NNCF,
            optimization_methods=OptimizationMethod.QUANTIZATION,
            optimization_objectives={},
            precision=[ModelPrecision.INT8],
            target_device=TargetDevice.CPU,
            performance_improvement={},
            model_size_reduction=1.,
            model_status=ModelStatus.NOT_READY)
        self.nncf_model.set_data('weights.pth', trained_model.get_data('weights.pth'))

        self.environment_for_nncf.model = self.nncf_model

        nncf_task_cls = get_task_class(nncf_task_class_impl_path)
        self.nncf_task = nncf_task_cls(task_environment=self.environment_for_nncf)

        logger.info('Run NNCF optimization')
        self.nncf_task.optimize(OptimizationType.NNCF,
                                dataset,
                                self.nncf_model,
                                OptimizationParameters())
        assert self.nncf_model.model_status == ModelStatus.SUCCESS, 'NNCF optimization was not successful'
        assert self.nncf_model.optimization_type == ModelOptimizationType.NNCF, 'Wrong optimization type'
        logger.info('NNCF optimization is finished')


    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'trained_model': results_prev_stages['training']['output_model'],
                'environment': results_prev_stages['training']['environment'],
        }

        self._run_ote_nncf(data_collector, **kwargs)
        results = {
                'nncf_task': self.nncf_task,
                'nncf_model': self.nncf_model,
                'nncf_environment': self.environment_for_nncf,
        }
        return results

class OTETestNNCFEvaluationAction(BaseOTETestAction):
    _name = 'nncf_evaluation'
    _with_validation = True

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_nncf_evaluation(self, data_collector,
                                dataset,
                                nncf_task,
                                nncf_model):
        logger.info('Begin evaluation of nncf model')
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(validation_dataset, nncf_task, nncf_model)
        data_collector.log_final_metric('evaluation_accuracy_nncf/' + score_name, score_value)
        logger.info('End evaluation of nncf model')
        return score_name, score_value

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'nncf'])

        kwargs = {
                'dataset': results_prev_stages['training']['dataset'],
                'nncf_task': results_prev_stages['nncf']['nncf_task'],
                'nncf_model': results_prev_stages['nncf']['nncf_model'],
        }

        score_name, score_value = self._run_ote_nncf_evaluation(data_collector, **kwargs)
        results = {
                'metrics': {
                    'accuracy': {
                        score_name: score_value
                    }
                }
        }
        return results

class OTETestNNCFExportAction(BaseOTETestAction):
    _name = 'nncf_export'

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_nncf_export(self, data_collector,
                             nncf_environment, dataset, nncf_task):
        logger.info('Begin export of nncf model')
        self.environment_nncf_export, self.nncf_exported_model = \
                run_export(nncf_environment, dataset, nncf_task, action_name=self.name)
        logger.info('End export of nncf model')

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'nncf'])

        kwargs = {
                'nncf_environment': results_prev_stages['nncf']['nncf_environment'],
                'dataset': results_prev_stages['training']['dataset'],
                'nncf_task': results_prev_stages['nncf']['nncf_task'],
        }

        self._run_ote_nncf_export(data_collector, **kwargs)
        results = {
                'environment': self.environment_nncf_export,
                'exported_model': self.nncf_exported_model,
        }
        return results

class OTETestNNCFExportEvaluationAction(BaseOTETestAction):
    _name = 'nncf_export_evaluation'
    _with_validation = True

    def __init__(self, subset=Subset.VALIDATION):
        self.subset = subset

    def _run_ote_nncf_export_evaluation(self, data_collector,
                                        model_template, dataset,
                                        nncf_environment_for_export, nncf_exported_model):
        logger.info('Begin evaluation of NNCF exported model')
        self.openvino_task = create_openvino_task(model_template, nncf_environment_for_export)
        validation_dataset = dataset.get_subset(self.subset)
        score_name, score_value = run_evaluation(validation_dataset, self.openvino_task, nncf_exported_model)
        data_collector.log_final_metric('evaluation_accuracy_nncf_exported/' + score_name, score_value)
        logger.info('End evaluation of NNCF exported model')
        return score_name, score_value

    def __call__(self, data_collector: DataCollector,
                 results_prev_stages: Optional[OrderedDict]=None):
        self._check_result_prev_stages(results_prev_stages, ['training', 'nncf_export'])

        kwargs = {
                'model_template': results_prev_stages['training']['model_template'],
                'dataset': results_prev_stages['training']['dataset'],
                'nncf_environment_for_export': results_prev_stages['nncf_export']['environment'],
                'nncf_exported_model': results_prev_stages['nncf_export']['exported_model'],
        }

        score_name, score_value = self._run_ote_nncf_export_evaluation(data_collector, **kwargs)
        results = {
                'metrics': {
                    'accuracy': {
                        score_name: score_value
                    }
                }
        }
        return results

def get_value_from_dict_by_dot_separated_address(struct, address):
    def _get(cur_struct, addr):
        assert isinstance(addr, list)
        if not addr:
            return cur_struct
        assert isinstance(cur_struct, dict)
        assert addr[0] in cur_struct, \
                f'Cannot find address {address} in struct {struct}: {addr} is absent in {cur_struct}'
        return _get(cur_struct[addr[0]], addr[1:])

    return _get(struct, address.split('.'))

def _check_mutual_exclusive_required_keys(struct, keys):
    assert isinstance(struct, dict)
    assert any(k in struct for k in keys), f'All of the keys {keys} are absent in the struct {struct}'
    num_occurences = sum(int(k in struct) for k in keys)
    assert num_occurences == 1, f'Wrong num occurences {num_occurences} of keys {keys} in the dict {struct}'

def validate_test_metrics(current_result, test_results_storage, current_test_expected_metrics):
    if current_test_expected_metrics == SKIP_VALIDATION():
        logger.info(f'current_test_expected_metrics is SKIP_VALIDATION, validation of test results is skipped')
        return
    if current_test_expected_metrics == ABSENT_EXPECTED_VALIDATION_METRICS():
        raise RuntimeError(f'Validation should be run, but expected metrics for the current test are absent')

    assert isinstance(current_test_expected_metrics, dict), \
            f'Wrong current test expected metric: {current_test_expected_metrics}'
    is_passed = True
    fail_reasons = []
    logger.info('Validation: begin')
    for k, v in current_test_expected_metrics.items():
        cur_res_addr = k
        cur_metric_requirements = v
        logger.info(f'Validation of test results: begin check {cur_res_addr}')
        current_metric = get_value_from_dict_by_dot_separated_address(current_result, cur_res_addr)
        _check_mutual_exclusive_required_keys(cur_metric_requirements, ['target_value', 'base'])
        if 'target_value' in cur_metric_requirements:
            target_value = float(cur_metric_requirements['target_value'])
        elif 'base' in cur_metric_requirements:
            base_metric_address = cur_metric_requirements['base']
            target_value = get_value_from_dict_by_dot_separated_address(test_results_storage, base_metric_address)
            target_value = float(target_value)
        else:
            raise RuntimeError(f'Wrong expected metrics {current_test_expected_metrics}')

        if 'max_drop' in cur_metric_requirements:
            target_value = target_value - float(cur_metric_requirements['max_drop'])

        if current_metric >= target_value:
            logger.info(f'Validation: passed: The metric {cur_res_addr} is greater or equal than the target value {target_value} '
                        f'({current_metric} vs {target_value})')
        else:
            cur_fail_reason = (f'Validation: failed: The metric {cur_res_addr} is less than the target value {target_value} '
                               f'({current_metric} vs {target_value})')
            logger.error(cur_fail_reason)
            fail_reasons.append(cur_fail_reason)
            is_passed = False
        logger.info(f'Validation of test results: end check {cur_res_addr}')

    logger.info(f'Validation: end, result={is_passed}')
    if not is_passed:
        fail_reasons = '\n'.join(fail_reasons)
        pytest.fail(f'Validation is failed:\n{fail_reasons}')

class OTETestStage:
    """
    OTETestStage -- auxiliary class that
    1. Allows to set up dependency between test stages: before the main action of a test stage is run, all the actions
       for the stages that are pointed in 'depends' list are called beforehand;
    2. Runs for each test stage its main action only once: the main action is run inside try-except clause, and
       2.1. if the action was executed without exceptions, a flag `was_processed` is set, the results of the action
            are kept, and the next time the stage is called no action is executed;
       2.2. if the action raised an exception, the exception is stored, the flag `was_processed` is set, and the next
            time the stage is called the exception is re-raised.
    """
    def __init__(self, action: BaseOTETestAction,
                 depends_stages: Optional[list]=None):
        self.was_processed = False
        self.stored_exception = None
        self.action = action
        self.depends_stages = depends_stages if depends_stages else []
        assert isinstance(self.depends_stages, list)
        assert all(isinstance(stage, OTETestStage) for stage in self.depends_stages)
        assert isinstance(self.action, BaseOTETestAction)

    @property
    def name(self):
        return self.action.name

    def was_ok(self):
        return self.was_processed and (self.stored_exception is None)

    def check_is_ok(self):
        if not self.was_processed:
            raise RuntimeError(f'Stage {self.name} was not run yet for this instance of OTEIntegrationTestCase')
        if self.was_ok():
            logger.info(f'The stage {self.name} was already processed SUCCESSFULLY')
            return

        logger.warning(f'In stage {self.name}: found that previous call of the stage '
                       'caused exception -- re-raising it')
        raise self.stored_exception

    def run_once(self, data_collector: DataCollector, test_results_storage: OrderedDict,
                 current_test_expected_metrics: Union[dict, str]):
        logger.info(f'Begin stage "{self.name}"')
        assert isinstance(test_results_storage, OrderedDict)
        logger.debug(f'For test stage "{self.name}": test_results_storage.keys = {list(test_results_storage.keys())}')

        for dep_stage in self.depends_stages:
            logger.debug(f'For test stage "{self.name}": Before running dep. stage "{dep_stage.name}"')
            dep_stage.run_once(data_collector, test_results_storage, current_test_expected_metrics=SKIP_VALIDATION())
            logger.debug(f'For test stage "{self.name}": After running dep. stage "{dep_stage.name}"')

        if self.was_processed:
            self.check_is_ok()
            logger.info(f'End stage "{self.name}"')
            return

        if self.name in test_results_storage:
            raise RuntimeError(f'Error: For test stage "{self.name}": '
                               f'another OTETestStage with name {self.name} has been run already')

        try:
            logger.info(f'For test stage "{self.name}": Before running main action')
            result_to_store = self.action(data_collector=data_collector,
                                          results_prev_stages=test_results_storage)
            logger.info(f'For test stage "{self.name}": After running main action')
            self.was_processed = True
            test_results_storage[self.name] = result_to_store
            logger.debug(f'For test stage "{self.name}": after addition test_results_storage.keys = '
                         f'{list(test_results_storage.keys())}')
        except Exception as e:
            logger.info(f'For test stage "{self.name}": After running action for stage {self.name} -- CAUGHT EXCEPTION:\n{e}')
            logger.info(f'End stage "{self.name}"')
            self.stored_exception = e
            self.was_processed = True
            raise e

        # This step is made outside the central try...except clause, since if the test was successful, but
        # the quality numbers were lower than expected, the result of the stage still may be re-used
        # in other stages.
        if self.action.with_validation and current_test_expected_metrics != SKIP_VALIDATION():
            validate_test_metrics(result_to_store, test_results_storage, current_test_expected_metrics)
        elif self.action.with_validation:
            logger.debug('The current_test_expected_metrics points that the validation should be skipped')
        logger.info(f'End stage "{self.name}"')

class OTEIntegrationTestCase:
    _TEST_STAGES = ('training', 'training_evaluation',
                   'export', 'export_evaluation',
                   'pot', 'pot_evaluation',
                   'nncf', 'nncf_evaluation',
                   'nncf_export', 'nncf_export_evaluation')

    @classmethod
    def get_list_of_test_stages(cls):
        return cls._TEST_STAGES

    def __init__(self, dataset_params: DatasetParameters, template_file_path: str,
                 num_training_iters: int, batch_size: int):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

        training_stage = OTETestStage(action=OTETestTrainingAction(self.dataset_params,
                                                                   self.template_file_path,
                                                                   self.num_training_iters,
                                                                   self.batch_size))
        training_evaluation_stage = OTETestStage(action=OTETestTrainingEvaluationAction(),
                                                 depends_stages=[training_stage])
        export_stage = OTETestStage(action=OTETestExportAction(),
                                    depends_stages=[training_stage])
        export_evaluation_stage = OTETestStage(action=OTETestExportEvaluationAction(),
                                               depends_stages=[export_stage])
        pot_stage = OTETestStage(action=OTETestPotAction(),
                                 depends_stages=[export_stage])
        pot_evaluation_stage = OTETestStage(action=OTETestPotEvaluationAction(),
                                            depends_stages=[pot_stage])
        nncf_stage = OTETestStage(action=OTETestNNCFAction(),
                                  depends_stages=[training_stage])
        nncf_evaluation_stage = OTETestStage(action=OTETestNNCFEvaluationAction(),
                                             depends_stages=[nncf_stage])
        nncf_export_stage = OTETestStage(action=OTETestNNCFExportAction(),
                                         depends_stages=[nncf_stage])
        nncf_export_evaluation_stage = OTETestStage(action=OTETestNNCFExportEvaluationAction(),
                                                    depends_stages=[nncf_export_stage])

        list_all_stages = [training_stage, training_evaluation_stage,
                           export_stage, export_evaluation_stage,
                           pot_stage, pot_evaluation_stage,
                           nncf_stage, nncf_evaluation_stage,
                           nncf_export_stage, nncf_export_evaluation_stage]

        self._stages = OrderedDict((stage.name, stage) for stage in list_all_stages)
        assert list(self._stages.keys()) == list(self._TEST_STAGES)

        # test results should be kept between stages
        self.test_results_storage = OrderedDict()

    def run_stage(self, stage_name, data_collector, current_test_expected_metrics):
        assert stage_name in self._TEST_STAGES, f'Wrong stage_name {stage_name}'
        self._stages[stage_name].run_once(data_collector, self.test_results_storage,
                                          current_test_expected_metrics)

# pytest magic
def pytest_generate_tests(metafunc):
    if metafunc.cls is None:
        return
    if not issubclass(metafunc.cls, TestOTEIntegration):
        return

    # It allows to filter by usecase
    usecase = metafunc.config.getoption('--test-usecase')

    argnames, argvalues, ids = metafunc.cls.get_list_of_tests(usecase)
    metafunc.parametrize(argnames, argvalues, ids=ids, scope='class')

class TestOTEIntegration:
    """
    The main class of running test in this file.
    It is responsible for all pytest magic.
    """
    PERFORMANCE_RESULTS = None # it is required for e2e system

    SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID = OrderedDict([
            ('test_stage', 'ACTION'),
            ('model_name', 'model'),
            ('dataset_name', 'dataset'),
            ('num_training_iters', 'num_iters'),
            ('batch_size', 'batch'),
            ('usecase', 'usecase'),
    ])

    # This tuple TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR describes test bunches'
    # fields that are important for creating OTEIntegrationTestCase instance.
    #
    # It is supposed that if for the next test these parameters are the same as
    # for the previous one, the result of operations in OTEIntegrationTestCase should
    # be kept and re-used.
    # See the fixture test_case_fx and the method _update_test_case_in_cache below.
    TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR = ('model_name',
                                              'dataset_name',
                                              'num_training_iters',
                                              'batch_size')

    DEFAULT_NUM_ITERS = 1
    DEFAULT_BATCH_SIZE = 2

    # Note that each test bunch describes a group of similar tests
    # If 'model_name' or 'dataset_name' are lists, cartesian product of tests will be run.
    # Each item may contain the following fields:
    #    * model_name
    #    * dataset_name
    #    * usecase
    #    * num_training_iters -- either integer value, or DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST,
    #                            or KEEP_CONFIG_FIELD_VALUE;
    #                            if None or absent, then DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST is used
    #    * batch_size -- either integer value, or DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST,
    #                    or KEEP_CONFIG_FIELD_VALUE;
    #                    if None or absent, then DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST is used
    test_bunches = [
            dict(
                model_name=[
                   'gen1_mobilenet_v2-2s_ssd-256x256',
                   'gen2_mobilenetV2_SSD',
                   'gen2_mobilenetV2_ATSS',
                   'gen2_resnet50_VFNet',
                ],
                dataset_name='dataset1_tiled_shortened_500_A',
                usecase='precommit',
            ),
            dict(
                model_name=[
                   'gen3_mobilenetV2_SSD',
                   'gen3_mobilenetV2_ATSS',
                   'gen3_resnet50_VFNet',
                ],
                dataset_name='dataset1_tiled_shortened_500_A',
                usecase='precommit',
            ),
            dict(
                model_name=[
                   'gen3_mobilenetV2_ATSS',
                ],
                dataset_name='bbcd',
                num_training_iters=KEEP_CONFIG_FIELD_VALUE(),
                batch_size=KEEP_CONFIG_FIELD_VALUE(),
                usecase='reallife',
            ),
    ]

    @staticmethod
    def _get_list_of_test_stages():
        return OTEIntegrationTestCase.get_list_of_test_stages()

    @classmethod
    def _fill_test_parameters_default_values(cls, test_parameters):
        def __set_default_if_required(key, default_val):
            val = test_parameters.get(key)
            if val is None or val == DEFAULT_FIELD_VALUE_FOR_USING_IN_TEST():
                test_parameters[key] = default_val

        __set_default_if_required('num_training_iters', cls.DEFAULT_NUM_ITERS)
        __set_default_if_required('batch_size',  cls.DEFAULT_BATCH_SIZE)

    @classmethod
    def _generate_test_id(cls, test_parameters):
        id_parts = (
                f'{short_par_name}-{test_parameters[par_name]}'
                for par_name, short_par_name in cls.SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID.items()
        )
        return ','.join(id_parts)

    @classmethod
    def get_list_of_tests(cls, usecase: Optional[str] = None):
        """
        The functions generates the lists of values for the tests from the field test_bunches of the class.

        The function returns two lists
        * argnames -- a tuple with names of the test parameters, at the moment it is
                      a one-element tuple with the parameter name "test_parameters"
        * argvalues -- list of tuples, each tuple has the same len as argname tuple,
                       at the moment it is a one-element tuple with the dict `test_parameters`
                       that stores the parameters of the test
        * ids -- list of strings with ids corresponding the parameters of the tests
                 each id is a string generated from the corresponding test_parameters
                 value -- see the functions _generate_test_id

        The lists argvalues and ids will have the same length.

        If the parameter `usecase` is set, it makes filtering by usecase field of test bunches.
        """
        test_bunches = cls.test_bunches
        assert all(isinstance(el, dict) for el in test_bunches)

        argnames = ('test_parameters',)
        argvalues = []
        ids = []
        for el in test_bunches:
            el_model_name = el.get('model_name')
            el_dataset_name = el.get('dataset_name')
            el_usecase = el.get('usecase')
            if usecase is not None and el_usecase != usecase:
                continue
            if isinstance(el_model_name, (list, tuple)):
                model_names = el_model_name
            else:
                model_names = [el_model_name]
            if isinstance(el_dataset_name, (list, tuple)):
                dataset_names = el_dataset_name
            else:
                dataset_names = [el_dataset_name]

            model_dataset_pairs = list(itertools.product(model_names, dataset_names))

            for m, d in model_dataset_pairs:
                for test_stage in cls._get_list_of_test_stages():
                    test_parameters = deepcopy(el)
                    test_parameters['test_stage'] = test_stage
                    test_parameters['model_name'] = m
                    test_parameters['dataset_name'] = d
                    cls._fill_test_parameters_default_values(test_parameters)
                    argvalues.append((test_parameters,))
                    ids.append(cls._generate_test_id(test_parameters))

        return argnames, argvalues, ids

    @pytest.fixture(scope='class')
    def cached_from_prev_test_fx(self):
        """
        This fixture is intended for storying the test case class OTEIntegrationTestCase and parameters
        for which the class is created.
        This object should be persistent between tests while the tests use the same parameters
        -- see the method _clean_cache_if_parameters_changed below that is used to clean
        the test case if the parameters are changed.
        """
        return dict()

    @staticmethod
    def _clean_cache_if_parameters_changed(cache, params_defining_cache):
        is_ok = True
        for k, v in params_defining_cache.items():
            is_ok = is_ok and (cache.get(k) == v)
        if is_ok:
            logger.info('TestOTEIntegration: parameters were not changed -- cache is kept')
            return

        for k in list(cache.keys()):
            del cache[k]
        for k, v in params_defining_cache.items():
            cache[k] = v
        logger.info('TestOTEIntegration: parameters were changed -- cache is cleaned')

    @classmethod
    def _update_test_case_in_cache(cls, cache,
                                   test_parameters,
                                   dataset_definitions, template_paths):
        """
        If the main parameters of the test differs w.r.t. the previous test,
        the cache will be cleared and new instance of OTEIntegrationTestCase will be created.
        Otherwise the previous instance of OTEIntegrationTestCase will be re-used
        """
        if dataset_definitions is None:
            pytest.skip('The parameter "--dataset-definitions" is not set')
        params_defining_cache = {k: test_parameters[k] for k in cls.TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR}

        assert '_test_case_' not in params_defining_cache, \
                'ERROR: parameters defining test behavior should not contain special key "_test_case_"'

        cls._clean_cache_if_parameters_changed(cache, params_defining_cache)

        if '_test_case_' not in cache:
            logger.info('TestOTEIntegration: creating OTEIntegrationTestCase')

            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']
            num_training_iters = test_parameters['num_training_iters']
            batch_size = test_parameters['batch_size']

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)
            template_path = _make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            cache['_test_case_'] = OTEIntegrationTestCase(dataset_params, template_path, num_training_iters, batch_size)

        return cache['_test_case_']

    @pytest.fixture
    def test_case_fx(self, current_test_parameters_fx, dataset_definitions_fx, template_paths_fx,
                     cached_from_prev_test_fx):
        """
        This fixture returns the test case class OTEIntegrationTestCase that should be used for the current test.
        Note that the cache from the fixture cached_from_prev_test_fx allows to store the instance of the class
        between the tests.
        If the main parameters used for this test are the same as the main parameters used for the previous test,
        the instance of the test case class will be kept and re-used. It is helpful for tests that can
        re-use the result of operations (model training, model optimization, etc) made for the previous tests,
        if these operations are time-consuming.
        If the main parameters used for this test differs w.r.t. the previous test, a new instance of
        TestOTEIntegration class will be created.
        """
        test_case = self._update_test_case_in_cache(cached_from_prev_test_fx,
                                                    current_test_parameters_fx,
                                                    dataset_definitions_fx, template_paths_fx)
        return test_case

    @pytest.fixture
    def data_collector_fx(self, request):
        setup = deepcopy(request.node.callspec.params)
        setup["environment_name"] = os.environ.get("TT_ENVIRONMENT_NAME", "no-env")
        setup["test_type"] = os.environ.get("TT_TEST_TYPE", "no-test-type")
        setup["scenario"] = "api"
        setup["test"] = request.node.name
        setup["subject"] = "custom-object-detection"
        setup["project"] = "ote"
        logger.info(f'creating DataCollector: setup=\n{pformat(setup, width=140)}')
        data_collector = DataCollector(name='TestOTEIntegration',
                                       setup=setup)
        with data_collector:
            logger.info('data_collector is created')
            yield data_collector
        logger.info('data_collector is released')

    @e2e_pytest_performance
    def test(self,
             test_parameters,
             test_case_fx, data_collector_fx,
             current_test_expected_metrics_fx):
        test_case_fx.run_stage(test_parameters['test_stage'], data_collector_fx,
                               current_test_expected_metrics_fx)
