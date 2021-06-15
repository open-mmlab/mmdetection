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

import copy
import io
import os
import shutil
import tempfile
import time
import torch
import warnings
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
from sc_sdk.configuration.configurable_parameters import ConfigurableParameter
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Dataset
from sc_sdk.entities.metrics import CurveMetric, LineChartInfo, MetricsGroup, Performance, ScoreMetric, InfoMetric, \
    VisualizationType, VisualizationInfo
from sc_sdk.entities.optimized_model import OptimizedModel, OpenVINOModel, Precision
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.entities.label_relations import ScoredLabel
from sc_sdk.entities.model import Model, NullModel
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.resultset import ResultSetEntity, ResultsetPurpose

from sc_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from sc_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from sc_sdk.usecases.repos import BinaryRepo
from sc_sdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from sc_sdk.usecases.tasks.interfaces.configurable_parameters_interface import IConfigurableParameters
from sc_sdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
from sc_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from sc_sdk.logging import logger_factory

from mmdet.apis import train_detector, get_root_logger, set_random_seed, single_gpu_test, \
    inference_detector, export_model
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

from mmcv.parallel import MMDataParallel
from mmcv.utils import Config
from mmcv.runner import load_checkpoint

from .configurable_parameters import MMDetectionParameters
from ..config import MMDetectionConfigManager, MMDetectionTaskType
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook, OTEProgressHook
from mmdet.integration.nncf import wrap_nncf_model
from mmdet.apis import get_fake_input

# The following imports are needed to register the custom datasets and hooks as modules in the
# mmdetection framework. They are not used directly in this file, but they have to be here for the registration to work
# from ...extension import *


logger = logger_factory.get_logger("OTEDetectionTask")
try:
    import logging
    from nncf.common.utils.logger import set_log_level
    set_log_level(logging.ERROR)
except ImportError:
    pass


class MMObjectDetectionTask(ImageDeepLearningTask, IConfigurableParameters, IModelOptimizer, IUnload):

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using OTEDetection.

        """
        logger.info(f"Loading OTEDetection task of type 'Detection' with task ID {task_environment.task_node.id}.")

        # Temp directory to store logs and model checkpoints
        self.scratch_space = tempfile.mkdtemp(prefix="ote-scratch-")
        logger.info(f"Scratch space created at {self.scratch_space}")
        self.mmdet_logger = get_root_logger(log_file=os.path.join(self.scratch_space, 'mmdet.log'))

        self.task_environment = task_environment

        # Initialize configuration manager to manage the configuration for the mmdetection framework, for this
        # particular task type and task environment
        self.config_manager = MMDetectionConfigManager(task_environment=task_environment,
                                                       task_type=MMDetectionTaskType.OBJECTDETECTION,
                                                       scratch_space=self.scratch_space)
        self.labels = task_environment.labels
        self.should_stop = False
        self.is_training = False

        # Model initialization.
        self.train_model = None
        self.inference_model = None
        self.compression_ctx = None
        self.learning_curves = defaultdict(OTELoggerHook.Curve)
        self.time_monitor = TimeMonitorCallback(0, 0, 0, 0)
        self.load_model(self.task_environment)

    def _create_model(self, config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: Model in training mode
        """
        model_cfg = copy.deepcopy(config.model)

        self.learning_curves = defaultdict(OTELoggerHook.Curve)

        init_from = config.get('init_from', None)
        if from_scratch:
            model_cfg.pretrained = None
            init_from = None
        logger.warning(init_from)
        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning('build detector')
            model = build_detector(model_cfg)
            # Load all weights.
            logger.warning('load checkpoint')
            load_checkpoint(model, init_from, map_location='cpu')
        else:
            logger.warning('build detector as is')
            model = build_detector(model_cfg)
        return model

    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:
        """ Analyzes a dataset using the latest inference model. """
        is_evaluation = analyse_parameters is not None and analyse_parameters.is_evaluation
        confidence_threshold = self._get_confidence(is_evaluation)

        batch_size = 1

        prediction_results = []
        if len(dataset) <= batch_size:
            # For small datasets,  just loop over the dataset_items and perform inference one by one
            for dataset_item in dataset:
                output = inference_detector(self.inference_model, dataset_item.numpy)
                prediction_results.append(output)
        else:
            # For larger datasets, build a data_loader to perform the analysis. This is much faster than one by one
            # inference.
            # First, update the dataset in the model config. The dataset is always set to the mmdetection test dataset.
            # FIXME. Why the dataset is always copied and re-created?
            self.inference_model.cfg.data.test.ote_dataset = dataset
            mm_test_dataset = build_dataset(copy.deepcopy(self.inference_model.cfg.data.test))
            # Use a single gpu for testing. Set in both mm_test_dataloader and prediction_model
            mm_test_dataloader = build_dataloader(mm_test_dataset, samples_per_gpu=batch_size, num_gpus=1, dist=False,
                                                  workers_per_gpu=self.config_manager.config.data.workers_per_gpu,
                                                  shuffle=False)
            # TODO. Support multi-gpu distributed setup.
            prediction_model = MMDataParallel(self.inference_model.cuda(self.config_manager.config.gpu_ids[0]),
                                              device_ids=self.config_manager.config.gpu_ids)
            prediction_results = single_gpu_test(prediction_model, mm_test_dataloader, show=False)

        # Loop over dataset again to assign predictions. Convert from MMdetection format to OTE format
        for dataset_item, output in zip(dataset, prediction_results):
            width = dataset_item.width
            height = dataset_item.height

            shapes = []
            for label_idx, detections in enumerate(output):
                for i in range(detections.shape[0]):
                    probability = float(detections[i, 4])
                    coords = detections[i, :4].astype(float).copy()
                    coords /= np.array([width, height, width, height], dtype=float)
                    coords = np.clip(coords, 0, 1)

                    if probability < confidence_threshold:
                        continue

                    assigned_label = [ScoredLabel(self.config_manager.config.labels[label_idx],
                                                  probability=probability)]
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    shapes.append(Box(x1=coords[0],
                                      y1=coords[1],
                                      x2=coords[2],
                                      y2=coords[3],
                                      labels=assigned_label))

            dataset_item.append_shapes(shapes)

        return dataset

    def load_model(self, task_environment: TaskEnvironment):
        """
        Load the model defined in the task environment. Both train_model and inference_model are loaded.
        This method is called when the task is loaded, and when the model architecture has changed in the configurable
        parameters of the task.
        Model creation without any pretrained weights for training from scratch is not handled here, that is done in
        the train method itself

        :param task_environment:
        """

        self.task_environment = task_environment
        model = task_environment.model

        if model != NullModel():
            # If a model has been trained and saved for the task already, create empty model and load weights here
            model_data = self._get_model_from_bytes(model.data)
            model_config = self.config_manager.config_from_string(model_data['config'])
            torch_model = self._create_model(config=model_config, from_scratch=True)

            try:
                torch_model.load_state_dict(model_data['state_dict'])
                logger.info(f"Loaded model weights from: {model.data_url}")
                logger.info(f"Model architecture: {model_config.model.type}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex

            self.inference_model = torch_model
            self.train_model = copy.deepcopy(self.inference_model)

        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file. These are ImageNet pretrained
            model_config = self.config_manager.config_copy
            logger.info(self.config_manager.config_to_string(model_config))
            torch_model = self._create_model(config=model_config, from_scratch=False)
            self.train_model = torch_model
            self.inference_model = copy.deepcopy(self.train_model)
            logger.info(f"No trained model in project yet. Created new model with {self.config_manager.model_name} "
                        f"architecture and ImageNet pretrained weights.")

        # Set the model configs. Inference always uses the config that came with the model, while training uses the
        # latest config in the config_manager
        self.inference_model.cfg = model_config
        self.train_model.cfg = self.config_manager.config_copy

        self.inference_model.eval()

    def _do_evaluation(self, model: torch.nn.Module, dataset: Dataset) -> float:
        """
        Performs evaluation of model using internal mAP metric.

        :return pretraining_performance (float): The performance score of the model
        """
        if model is not None:
            logger.info("Evaluating model.")
            # Build the dataset with the correct data configuration. Config has to come from the model, not the
            # config_manager, because architecture might have changed
            model = self.config_manager.update_dataset_subsets(dataset, model)
            mm_val_dataset = build_dataset(model.cfg.data.test)
            batch_size = 1
            mm_val_dataloader = build_dataloader(mm_val_dataset,
                                                 samples_per_gpu=batch_size,
                                                 workers_per_gpu=self.config_manager.config.data.workers_per_gpu,
                                                 num_gpus=1,
                                                 dist=False,
                                                 shuffle=False)
            eval_model = MMDataParallel(model.cuda(self.config_manager.config.gpu_ids[0]),
                                        device_ids=self.config_manager.config.gpu_ids)
            # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
            eval_predictions = single_gpu_test(eval_model, mm_val_dataloader, show=False)
            eval_results = mm_val_dataset.evaluate(eval_predictions, metric='mAP')
            pretraining_performance = eval_results['mAP']
        else:
            pretraining_performance = 0.0
        logger.info(f"Model performance: mAP = {pretraining_performance}")
        return pretraining_performance

    def _persist_inference_model(self, dataset: Dataset, performance: Performance, training_duration: float):
        model_data = self._get_model_bytes(self.inference_model)
        model = Model(project=self.task_environment.project,
                      task_node=self.task_environment.task_node,
                      configuration=self.task_environment.get_model_configuration(),
                      data=model_data,
                      tags=None,
                      performance=performance,
                      train_dataset=dataset,
                      training_duration=training_duration)
        self.task_environment.model = model

    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        """ Trains a model on a dataset """

        # FIXME. Looks like implementation is not intuitive here. This is what it does now:
        # 1. Build the dataset in a proper format for training. (fine).
        # 2. Overrides training model, if there is need to reset the weights and train from scratch. (fine).
        # 3. Evaluates performance before training. (could be done at the upper level via analyze/performace interface).
        # 4. Do training. (fine).
        # 5. Evaluate the best obtained model and check if it improved. (see 3).
        # 6. If model improved (over what?), replace it in task_environment. (this is a side-effect which, IMO, better be ommited here).

        if self.train_model is None:
            raise ValueError("Training model is not initialized. Please load the trainable model to the task before training.")

        # Create a directory to store model checkpoints for this training round. Also writes the model config to a file
        # 'config.py' in that directory, for debugging purposes.
        self.config_manager.prepare_work_dir(self.scratch_space)

        # Create new train_model if training from scratch.
        old_train_model = copy.deepcopy(self.train_model)
        if train_parameters is not None and train_parameters.train_on_empty_model:
            logger.info("Training from scratch, created new model")
            self.train_model = self._create_model(config=self.config_manager.config, from_scratch=True)

        # Evaluate model performance before training.
        logger.warning('PREEVALUATION')
        initial_performance = self._do_evaluation(self.inference_model, dataset)

        # Check for stop signal between pre-eval and training. If training is cancelled at this point,
        # old_train_model should be restored.
        if self.should_stop:
            self.should_stop = False
            logger.info('Training cancelled.')
            self.train_model = old_train_model
            return self.task_environment.model

        # Create inference model as a copy of a train one.
        # FIXME.
        self.train_model.cfg = self.config_manager.config_copy
        inference_model = copy.deepcopy(self.train_model)
        inference_model.eval()

        self.config_manager.update_dataset_subsets(dataset)
        mm_train_dataset = build_dataset(self.config_manager.config.data.train)
        config = self.config_manager.config_copy
        config.log_config.hooks = [{'type': 'OTELoggerHook', 'curves': self.learning_curves}]
        if config.get('custom_hooks', None) is None:
            config.custom_hooks = []
        self.time_monitor = TimeMonitorCallback(0, 0, 0, 0) # It will be initialized properly inside the OTEProgressHook before training.
        config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': self.time_monitor, 'verbose': True})

        # Train the model. Training modifies mmdet config in place, so make a deepcopy
        self.is_training = True
        start_train_time = time.time()
        train_detector(model=self.train_model,
                       dataset=[mm_train_dataset],
                       cfg=config,
                       validate=True)
        training_duration = time.time() - start_train_time

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self.should_stop:
            self.should_stop = False
            logger.info('Training cancelled.')
            self.train_model = old_train_model
            return self.task_environment.model

        # Load the best weights and check if model has improved
        training_metrics = self._generate_training_metrics_group()
        best_checkpoint_path = os.path.join(self.train_model.cfg.work_dir, 'latest.pth')
        if self.train_model.cfg.get('nncf_config'):
            self.train_model = None
            cfg = inference_model.cfg
            cfg.load_from = best_checkpoint_path
            self.compression_ctx, inference_model = wrap_nncf_model(inference_model, cfg, None, get_fake_input)
            cfg.load_from = None
            inference_model.cfg = cfg
        else:
            best_checkpoint = torch.load(best_checkpoint_path)
            inference_model.load_state_dict(best_checkpoint['state_dict'])

        # Evaluate model performance after training.
        logger.warning('POSTEVALUATION')
        final_performance = self._do_evaluation(inference_model, dataset)
        improved = final_performance > initial_performance

        # Return a new model if model has improved, or there is no model yet.
        if improved or isinstance(self.task_environment.model, NullModel):
            if improved:
                logger.info("Training finished, and it has an improved model")
            else:
                logger.info("First training round, saving the model.")
            # Add mAP metric and loss curves
            performance = Performance(score=ScoreMetric(value=final_performance, name="mAP"),
                                      dashboard_metrics=training_metrics)
            logger.info('FINAL MODEL PERFORMANCE\n' + str(performance))
            self.inference_model = inference_model
            self._persist_inference_model(dataset, performance, training_duration)
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
            # Restore old training model if training from scratch and not improved
            self.train_model = old_train_model

        self.is_training = False
        return self.task_environment.model

    def get_training_progress(self) -> float:
        """
        Calculate the progress of the current training

        :return: training progress in percent
        """
        return self.time_monitor.get_progress()

    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self.should_stop = True
        stop_training_filepath = os.path.join(self.config_manager.config.work_dir, '.stop_training')
        open(stop_training_filepath, 'a').close()

    def compute_performance(self, resultset: ResultSetEntity) -> Performance:
        """ Computes performance on a resultset """
        params = self.get_configurable_parameters(self.task_environment)

        result_based_confidence_threshold = params.postprocessing.result_based_confidence_threshold.value

        f_measure_metrics = MetricsHelper.compute_f_measure(resultset,
                                                            result_based_confidence_threshold,
                                                            False,
                                                            False)

        if resultset.purpose is ResultsetPurpose.EVALUATION:
            # only set configurable params based on validation result set
            if result_based_confidence_threshold:
                best_confidence_threshold = f_measure_metrics.best_confidence_threshold
                if best_confidence_threshold is not None:
                    logger.info(f"Setting confidence_threshold to " f"{best_confidence_threshold.value} based on results")
                    params.postprocessing.confidence_threshold.value = best_confidence_threshold.value
                else:
                    raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")

            self.task_environment.set_configurable_parameters(params)

        logger.info(f"F-measure after evaluation: {f_measure_metrics.f_measure.value}")

        return f_measure_metrics.get_performance()

    def _generate_training_metrics_group(self) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self.config_manager.model_name)
        visualization_info_architecture = VisualizationInfo(name="Model architecture",
                                                            visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[architecture],
                                   visualization_info=visualization_info_architecture))

        # Learning rate schedule
        learning_rate_schedule = InfoMetric(
            name='Model architecture',
            value=self.config_manager.get_lr_schedule_friendly_name(self.train_model.cfg.lr_config.policy)
        )
        visualization_info_lr_schedule = VisualizationInfo(name="Learning rate schedule",
                                                           visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[learning_rate_schedule],
                                   visualization_info=visualization_info_lr_schedule))

        # Learning curves
        for key, curve in self.learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output

    def _get_model_bytes(self, model: torch.nn.Module = None) -> bytes:
        """
        Returns the data of the current model. We store both the state_dict and the configuration to make the model
        self-contained.

        :return: {'state_dict': data of current model in bytes, 'config': mmdetection config string used for training}
        """
        buffer = io.BytesIO()
        if model is None:
            model = self.train_model
        config_str = self.config_manager.config_to_string(model.cfg)
        meta = model.cfg.checkpoint_config.get('meta', {})
        torch.save({'state_dict': model.state_dict(), 'config': config_str, 'meta': meta},
                   buffer)
        return bytes(buffer.getbuffer())

    def _get_model_from_bytes(self, blob: bytes) -> dict:
        buffer = io.BytesIO(blob)
        return torch.load(buffer)

    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> MMDetectionParameters:
        """
        Returns the configurable parameters.

        :param task_environment: Current task environment
        :return: Instance of MMDetectionParameters
        """
        return task_environment.get_configurable_parameters(instance_of=MMDetectionParameters)

    @staticmethod
    def apply_template_configurable_parameters(params: MMDetectionParameters, template: dict):

        def xset(obj: ConfigurableParameter, d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    xset(obj[k], v)
                else:
                    if hasattr(getattr(obj, k), 'value'):
                        getattr(obj, k).value = type(getattr(obj, k).value)(v)
                    else:
                        setattr(obj, k, v)

        hyper_params = template['hyper_parameters']['params']
        xset(params, hyper_params)

        params.algo_backend.model_name.value = template['name']
        return params

    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        """
        Called when the user changes the configurable parameters in the UI.

        :param task_environment: New task environment with updated configurable parameters
        """
        new_conf_params = self.get_configurable_parameters(task_environment)
        self.task_environment = task_environment
        self.config_manager.update_project_configuration(new_conf_params)

    def _get_confidence(self, is_evaluation: bool) -> Tuple[float, float, bool]:
        """
        Retrieves the thresholds for confidence from the configurable parameters. If
        is_evaluation is True, the confidence threshold is set to 0 in order to compute optimum values
        for the thresholds. Also returns whether or not to perform nms across objects of different classes.

        :param is_evaluation: bool, True in case analysis is requested for evaluation

        :return confidence_threshold: float, threshold for prediction confidence
        """
        conf_params = self.get_configurable_parameters(self.task_environment)
        confidence_threshold = conf_params.postprocessing.confidence_threshold.value
        result_based_confidence_threshold = conf_params.postprocessing.result_based_confidence_threshold.value
        if is_evaluation:
            if result_based_confidence_threshold:
                confidence_threshold = 0.0
        return confidence_threshold

    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container

        :return bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        is_in_docker = False
        if os.path.isfile(path):
            with open(path) as f:
                is_in_docker = is_in_docker or any('docker' in line for line in f)
        is_in_docker = is_in_docker or os.path.exists('/.dockerenv')
        return is_in_docker

    def unload(self):
        """
        Unload the task
        """
        self._delete_scratch_space()
        if self._is_docker():
            logger.warning(
                "Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes
            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(f"Done unloading. "
                           f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory")

    def optimize_loaded_model(self) -> List[OptimizedModel]:
        """
        Create list of optimized models. Currently only OpenVINO models are supported.
        """
        optimized_models = [self._optimize_model_openvino()]
        return optimized_models

    def _optimize_model_openvino(self, params: dict = {}) -> OpenVINOModel:
        optimized_model_precision = Precision.FP32

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "otedet")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                if self.compression_ctx:
                    self.compression_ctx.prepare_for_export()
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings("ignore", category=TracerWarning)
                export_model(self.inference_model, tempdir, target='openvino', precision=optimized_model_precision.name)
                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                openvino_bin_url = BinaryRepo(self.task_environment.project).save_file_at_path(
                    os.path.join(tempdir, bin_file), "optimized_models")
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                openvino_xml_url = BinaryRepo(self.task_environment.project).save_file_at_path(
                    os.path.join(tempdir, xml_file), "optimized_models")
            except Exception as ex:
                raise RuntimeError("Optimization was unsuccessful.") from ex

        return OpenVINOModel(model=self.task_environment.model,
                             openvino_bin_url=openvino_bin_url,
                             openvino_xml_url=openvino_xml_url,
                             precision=optimized_model_precision)

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space, ignore_errors=False)
