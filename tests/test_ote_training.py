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
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType

from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import get_task_class
from mmdet.apis.ote.extension.datasets.data_utils import load_dataset_items_coco_format

logger = logging.getLogger(__name__)

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

class OTETrainingImpl:
    def __init__(self, dataset_params: DatasetParameters, template_file_path: str,
                 num_training_iters: int, batch_size: int):
        self.dataset_params = dataset_params
        self.template_file_path = template_file_path
        self.num_training_iters = num_training_iters
        self.batch_size = batch_size

        self.template = None
        self.environment = None
        self.task = None
        self.output_model = None
        self.evaluation_performance = None
        self.environment_for_export = None
        self.exported_model = None
        self.openvino_task = None
        self.evaluation_performance_exported = None

        self.was_training_run = False
        self.stored_exception_training = None
        self.was_export_run = False
        self.stored_exception_export = None

        self.copy_hyperparams = None

    @staticmethod
    def _create_environment_and_task(params, labels_schema, model_template):
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)
        logger.info('Create base Task')
        task_impl_path = model_template.entrypoints.base
        task_cls = get_task_class(task_impl_path)
        task = task_cls(task_environment=environment)
        return environment, task

    def _run_ote_training(self):
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
        params.learning_parameters.num_iters = self.num_training_iters
        if self.num_training_iters < 20:
            num_checkpoints = 2
        elif self.num_training_iters < 1000:
            num_checkpoints = 10
        else:
            num_checkpoints = 30

        params.learning_parameters.batch_size = self.batch_size

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

        # To check that the training performance is valid
        # and may be converted to score name and value
        self.get_training_performance_as_score_name_value()
        logger.info(f'performance={self.output_model.performance}')

    def get_training_performance_as_score_name_value(self):
        training_performance = getattr(self.output_model, 'performance', None)
        if training_performance is None:
            raise RuntimeError('Cannot get training performance')
        return performance_to_score_name_value(training_performance)

    def run_ote_training_once(self, data_collector):
        if self.was_training_run and self.stored_exception_training:
            logger.warning('In function run_ote_training_once: found that previous call of the function '
                           'caused exception -- re-raising it')
            raise self.stored_exception_training

        if not self.was_training_run:
            try:
                self._run_ote_training()
                self.was_training_run = True
            except Exception as e:
                self.stored_exception_training = e
                self.was_training_run = True
                raise e

        score_name, score_value = self.get_training_performance_as_score_name_value()
        data_collector.log_final_metric('training_performance/' + score_name, score_value)

        hyperparams_dict = convert_hyperparams_to_dict(self.copy_hyperparams)
        for k, v in hyperparams_dict.items():
            data_collector.update_metadata(k, v)

        return score_name, score_value

    def run_ote_evaluation(self, data_collector, subset=Subset.VALIDATION):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        logger.debug('Get predictions on the validation set')
        validation_dataset = self.dataset.get_subset(subset)
        self.predicted_validation_dataset = self.task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        self.resultset = ResultSetEntity(
            model=self.output_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=self.predicted_validation_dataset,
        )
        logger.debug('Estimate quality on validation set')
        self.task.evaluate(self.resultset)
        self.evaluation_performance = self.resultset.performance
        logger.info(f'performance={self.evaluation_performance}')
        score_name, score_value = performance_to_score_name_value(self.evaluation_performance)
        data_collector.log_final_metric('evaluation_performance/' + score_name, score_value)
        return self.evaluation_performance

    def _run_ote_export(self, data_collector):
        logger.debug('Copy environment for evaluation exported model')

        # Warning: Note that this code does not make copy of self.environment -- it works only if
        # the task operations use the environment in read-only mode.
        # At the moment this is true, but if it is changed in the future, side effects may cause errors.
        self.environment_for_export = self.environment

        logger.debug('Create exported model')
        self.exported_model = ModelEntity(
            self.dataset,
            self.environment_for_export.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        logger.debug('Run export')
        self.task.export(ExportType.OPENVINO, self.exported_model)
        logger.debug('Set exported model into environment for export')
        self.environment_for_export.model = self.exported_model

    def run_ote_export_once(self, data_collector):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if self.was_export_run and self.stored_exception_export:
            logger.warning('In function run_ote_export_once: found that previous call of the function '
                           'caused exception -- re-raising it')
            raise self.stored_exception_export

        if not self.was_export_run:
            try:
                self._run_ote_export(data_collector)
                self.was_export_run = True
            except Exception as e:
                self.stored_exception_export = e
                self.was_export_run = True
                raise e

    def run_ote_evaluation_exported(self, data_collector, subset=Subset.VALIDATION):
        if not self.was_training_run:
            raise RuntimeError('Training was not run for the OTETrainingImpl instance')
        if self.stored_exception_training:
            raise RuntimeError('Training was not successful for the OTETrainingImpl instance')
        if not self.was_export_run:
            raise RuntimeError('Export was not run for the OTETrainingImpl instance')
        if self.stored_exception_export:
            raise RuntimeError('Export was not successful for the OTETrainingImpl instance')

        logger.debug('Create OpenVINO Task')
        openvino_task_impl_path = self.model_template.entrypoints.openvino
        openvino_task_cls = get_task_class(openvino_task_impl_path)
        self.openvino_task = openvino_task_cls(self.environment_for_export)

        logger.debug('Get predictions on the validation set')
        validation_dataset = self.dataset.get_subset(subset)
        self.predicted_validation_dataset_exp = self.openvino_task.infer(
            validation_dataset.with_empty_annotations(),
            InferenceParameters(is_evaluation=True))
        self.resultset_exp = ResultSetEntity(
            model=self.exported_model,
            ground_truth_dataset=validation_dataset,
            prediction_dataset=self.predicted_validation_dataset_exp,
        )
        logger.debug('Estimate quality on validation set')
        self.openvino_task.evaluate(self.resultset_exp)
        self.evaluation_performance_exported = self.resultset_exp.performance

        logger.info(f'performance exported={self.evaluation_performance_exported}')
        score_name, score_value = performance_to_score_name_value(self.evaluation_performance_exported)
        data_collector.log_final_metric('evaluation_performance_exported/' + score_name, score_value)
        return self.evaluation_performance_exported

# pytest magic
def pytest_generate_tests(metafunc):
    if metafunc.cls is None:
        return
    if not issubclass(metafunc.cls, TestOTETraining):
        return

    # It allows to filter by usecase
    usecase = metafunc.config.getoption('--test-usecase')

    argnames, argvalues, ids = metafunc.cls.get_list_of_tests(usecase)
    metafunc.parametrize(argnames, argvalues, ids=ids, scope='class')

class TestOTETraining:
    PERFORMANCE_RESULTS = None # it is required for e2e system

    DEFAULT_NUM_ITERS = 1
    DEFAULT_BATCH_SIZE = 2
    SHORT_TEST_PARAMETERS_NAMES_FOR_GENERATING_ID = OrderedDict([
            ('model_name', 'model'),
            ('dataset_name', 'dataset'),
            ('num_training_iters', 'num_iters'),
            ('batch_size', 'batch'),
            ('usecase', 'usecase'),
    ])

    # This tuple TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR describes test bunches'
    # fields that are important for creating OTETrainingImpl instance.
    #
    # It is supposed that if for the next test these parameters are the same as
    # for the previous one, the result of operations in OTETrainingImpl should
    # be kept and re-used.
    # See the fixture impl_fx and the method _update_impl_in_cache below.
    TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR = ('model_name',
                                              'dataset_name',
                                              'num_training_iters',
                                              'batch_size')

    # Note that each test bunch describes a group of similar tests
    # If 'model_name' or 'dataset_name' are lists, cartesian product of tests will be run.
    test_bunches = [
#           dict(
#               model_name=[
#                   'face-detection-0200',
#                   'face-detection-0202',
#                   'face-detection-0204',
#                   'face-detection-0205',
#                   'face-detection-0206',
#                   'face-detection-0207',
#               ],
#               dataset_name='airport_faces',
#               usecase='precommit',
#           ),
#           dict(
#               model_name=[
#                   'horizontal-text-detection-0001',
#               ],
#               dataset_name='horizontal_text_detection',
#               usecase='precommit',
#           ),
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
#            dict(
#                model_name=[
#                    'person-detection-0200',
#                    'person-detection-0201',
#                    'person-detection-0202',
#                    'person-detection-0203'
#                ],
#                dataset_name='airport_person',
#                usecase='precommit',
#            ),
#            dict(
#                model_name=[
#                    'person-vehicle-bike-detection-2000',
#                    'person-vehicle-bike-detection-2001',
#                    'person-vehicle-bike-detection-2002',
#                    'person-vehicle-bike-detection-2003',
#                    'person-vehicle-bike-detection-2004'
#                ],
#                dataset_name='airport_example',
#                usecase='precommit',
#            ),
#            dict(
#                model_name=[
#                    'vehicle-detection-0200',
#                    'vehicle-detection-0201',
#                    'vehicle-detection-0202',
#                    'vehicle-detection-0203',
#                ],
#                dataset_name='vehicle_detection',
#                usecase='precommit',
#            ),
    ]


    @classmethod
    def _fill_test_parameters_default_values(cls, test_parameters):
        test_parameters['num_training_iters'] = test_parameters.get('num_training_iters', cls.DEFAULT_NUM_ITERS)
        test_parameters['batch_size'] = test_parameters.get('batch_size', cls.DEFAULT_BATCH_SIZE)

    @classmethod
    def _generate_test_id(cls, test_parameters):
        id_parts = (
                f'{short_par_name}={test_parameters[par_name]}'
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
                test_parameters = deepcopy(el)
                test_parameters['model_name'] = m
                test_parameters['dataset_name'] = d
                cls._fill_test_parameters_default_values(test_parameters)
                argvalues.append((test_parameters,))
                ids.append(cls._generate_test_id(test_parameters))

        return argnames, argvalues, ids

    @pytest.fixture(scope='class')
    def cached_from_prev_test_fx(self):
        """
        This fixture is intended for storying the impl class OTETrainingImpl and parameters
        for which the class is created.
        This object should be persistent between tests while the tests use the same parameters
        -- see the method _clean_cache_if_parameters_changed below that is used to clean
        the impl if the parameters are changed.
        """
        return dict()

    @staticmethod
    def _clean_cache_if_parameters_changed(cache, params_defining_cache):
        is_ok = True
        for k, v in params_defining_cache.items():
            is_ok = is_ok and (cache.get(k) == v)
        if is_ok:
            logger.info('TestOTETraining: parameters were not changed -- cache is kept')
            return

        for k in list(cache.keys()):
            del cache[k]
        for k, v in params_defining_cache.items():
            cache[k] = v
        logger.info('TestOTETraining: parameters were changed -- cache is cleaned')

    @classmethod
    def _update_impl_in_cache(cls, cache,
                              test_parameters,
                              dataset_definitions, template_paths):
        """
        If the main parameters of the test differs w.r.t. the previous test,
        the cache will be cleared and new instance of OTETrainingImpl will be created.
        Otherwise the previous instance of OTETrainingImpl will be re-used
        """
        if dataset_definitions is None:
            pytest.skip('The parameter "--dataset-definitions" is not set')
        params_defining_cache = {k: test_parameters[k] for k in cls.TEST_PARAMETERS_DEFINING_IMPL_BEHAVIOR}

        assert '_impl_' not in params_defining_cache, \
                'ERROR: parameters defining test behavior should not contain special key "_impl_"'

        cls._clean_cache_if_parameters_changed(cache, params_defining_cache)

        if '_impl_' not in cache:
            logger.info('TestOTETraining: creating OTETrainingImpl')

            model_name = test_parameters['model_name']
            dataset_name = test_parameters['dataset_name']
            num_training_iters = int(test_parameters['num_training_iters'])
            batch_size = int(test_parameters['batch_size'])

            dataset_params = _get_dataset_params_from_dataset_definitions(dataset_definitions, dataset_name)
            template_path = _make_path_be_abs(template_paths[model_name], template_paths[ROOT_PATH_KEY])

            cache['_impl_'] = OTETrainingImpl(dataset_params, template_path, num_training_iters, batch_size)

        return cache['_impl_']

    @pytest.fixture
    def impl_fx(self, request, dataset_definitions_fx, template_paths_fx,
                cached_from_prev_test_fx):
        """
        This fixture returns the impl class OTETrainingImpl that should be used for the current test.
        Note that the cache from the fixture cached_from_prev_test_fx allows to store the instance of the class
        between the tests.
        If the main parameters used for this test are the same as the main parameters used for the previous test,
        the instance of the implementation class will be kept and re-used. It is helpful for tests that can
        re-use the result of operations (model training, model optimization, etc) made for the previous tests,
        if these operations are time-consuming.
        If the main parameters used for this test differs w.r.t. the previous test, a new instance of TestOTETraining
        class will be created.
        """
        cur_request_parameters = deepcopy(request.node.callspec.params)
        if 'test_parameters' not in cur_request_parameters:
            raise RuntimeError(f'Test {request.node.name} should be parametrized by parameter "test_parameters"')

        test_parameters = cur_request_parameters['test_parameters']
        impl = self._update_impl_in_cache(cached_from_prev_test_fx,
                                          test_parameters,
                                          dataset_definitions_fx, template_paths_fx)
        return impl

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
        data_collector = DataCollector(name='TestOTETraining',
                                       setup=setup)
        with data_collector:
            logger.info('data_collector is created')
            yield data_collector
        logger.info('data_collector is released')

    @e2e_pytest_performance
    def test_ote_01_training(self,
                             test_parameters, # is required for impl_fx magic
                             impl_fx, data_collector_fx):
        impl_fx.run_ote_training_once(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_02_evaluation(self,
                               test_parameters, # is required for impl_fx magic
                               impl_fx, data_collector_fx):
        impl_fx.run_ote_training_once(data_collector_fx)
        impl_fx.run_ote_evaluation(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_03_export(self,
                           test_parameters, # is required for impl_fx magic
                           impl_fx, data_collector_fx):
        impl_fx.run_ote_training_once(data_collector_fx)
        impl_fx.run_ote_export_once(data_collector_fx)

    @e2e_pytest_performance
    def test_ote_04_evaluation_exported(self,
                                        test_parameters, # is required for impl_fx magic
                                        impl_fx, data_collector_fx):
        impl_fx.run_ote_training_once(data_collector_fx)
        impl_fx.run_ote_export_once(data_collector_fx)
        impl_fx.run_ote_evaluation_exported(data_collector_fx)
