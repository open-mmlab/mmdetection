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
from sc_sdk.configuration import cfg_helper
from sc_sdk.configuration.helper.utils import ids_to_strings
from sc_sdk.entities.annotation import Annotation
from sc_sdk.entities.analyse_parameters import AnalyseParameters
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.metrics import CurveMetric, LineChartInfo, MetricsGroup, Performance, ScoreMetric, InfoMetric, \
    VisualizationType, VisualizationInfo
# This one breaks cyclic imports chain.
from sc_sdk.usecases.repos import BinaryRepo
from sc_sdk.entities.optimized_model import OptimizedModel, ModelPrecision
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.entities.train_parameters import TrainParameters
from sc_sdk.entities.label import ScoredLabel
from sc_sdk.entities.model import Model, ModelStatus, NullModel
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.resultset import ResultSetEntity, ResultsetPurpose
from sc_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from sc_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from sc_sdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from sc_sdk.usecases.tasks.interfaces.export_task import IExportTask, ExportType
from sc_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from sc_sdk.logging import logger_factory

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.apis import train_detector, get_root_logger, set_random_seed, single_gpu_test, export_model
from mmdet.apis.ote.apis.detection.configuration import ObjectDetectionConfig
from mmdet.apis.ote.apis.detection.config_utils import (patch_config, set_hyperparams, prepare_for_training,
    prepare_for_testing, config_from_string, config_to_string)
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


logger = logger_factory.get_logger("OTEDetectionTask")


class MMObjectDetectionTask(ImageDeepLearningTask, IExportTask, IUnload):

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using OTEDetection.

        """
        logger.info(f"Loading OTEDetection task.")
        self.scratch_space = tempfile.mkdtemp(prefix="ote-scratch-")
        logger.info(f"Scratch space created at {self.scratch_space}")
        self.mmdet_logger = get_root_logger(log_file=os.path.join(self.scratch_space, 'mmdet.log'))

        self.task_environment = task_environment

        self.should_stop = False
        self.is_training = False
        self.time_monitor = None

        # Model initialization.
        self.train_model = None
        self.inference_model = None
        self.load_model(self.task_environment)
        self.update_configurable_parameters(self.task_environment)


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
        hyperparams = task_environment.get_configurable_parameters(ObjectDetectionConfig)
        self.model_name = hyperparams.algo_backend.model_name
        self.labels = task_environment.get_labels(False)

        if model != NullModel():
            # If a model has been trained and saved for the task already, create empty model and load weights here
            model_data = self._get_model_from_bytes(model.get_data("weights.pth"))
            self.config = config_from_string(model_data['mmdet_config'])
            self.inference_model = self._create_model(config=self.config, from_scratch=True)

            try:
                self.inference_model.load_state_dict(model_data['model'])
                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self.model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex

            self.train_model = copy.deepcopy(self.inference_model)

        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            template_file_path = hyperparams.algo_backend.template
            base_dir = os.path.abspath(os.path.dirname(template_file_path))
            config_file_path = os.path.join(base_dir, hyperparams.algo_backend.model)
            self.config = Config.fromfile(config_file_path)
            patch_config(self.config, self.scratch_space, self.labels, random_seed=42)
            self.inference_model = self._create_model(config=self.config, from_scratch=False)
            self.train_model = copy.deepcopy(self.inference_model)
            logger.info(f"No trained model in project yet. Created new model with '{self.model_name}' "
                        f"architecture and general-purpose pretrained weights.")

        # Set the model configs. Inference always uses the config that came with the model, while training uses the
        # latest config in the config_manager
        self.train_model.cfg = copy.deepcopy(self.config)
        self.inference_model.cfg = copy.deepcopy(self.config)
        self.inference_model.eval()


    def _create_model(self, config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: Model in training mode
        """
        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get('load_from', None)
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
            logger.warning('build detector')
            model = build_detector(model_cfg)
        return model


    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:
        """ Analyzes a dataset using the latest inference model. """
        is_evaluation = analyse_parameters is not None and analyse_parameters.is_evaluation
        confidence_threshold = self._get_confidence(is_evaluation)
        logger.info(f'Confidence threshold {confidence_threshold}')

        prediction_results, _ = self._do_inference(self.inference_model, dataset, False)

        # Loop over dataset again to assign predictions. Convert from MMDetection format to OTE format
        from tqdm import tqdm
        for dataset_item, output in tqdm(zip(dataset, prediction_results)):
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

                    assigned_label = [ScoredLabel(self.labels[label_idx],
                                                  probability=probability)]
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    shapes.append(Annotation(
                        Box(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label))

            dataset_item.append_annotations(shapes)
            # print(dataset_item)

        return dataset


    def _do_inference(self, model: torch.nn.Module, dataset: Dataset,
                      eval: Optional[bool] = False, metric_name: Optional[str] = 'mAP') -> Tuple[List, float]:
        test_config = prepare_for_testing(model.cfg, dataset)
        mm_val_dataset = build_dataset(test_config.data.test)
        batch_size = 1
        mm_val_dataloader = build_dataloader(mm_val_dataset,
                                                samples_per_gpu=batch_size,
                                                workers_per_gpu=test_config.data.workers_per_gpu,
                                                num_gpus=1,
                                                dist=False,
                                                shuffle=False)
        eval_model = MMDataParallel(model.cuda(test_config.gpu_ids[0]),
                                    device_ids=test_config.gpu_ids)
        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        eval_predictions = single_gpu_test(eval_model, mm_val_dataloader, show=False)

        metric = None
        if eval:
            metric = mm_val_dataset.evaluate(eval_predictions, metric=metric_name)[metric_name]
        return eval_predictions, metric


    def compute_performance(self, resultset: ResultSetEntity) -> Performance:
        """ Computes performance on a resultset """
        params = self.get_configurable_parameters(self.task_environment)

        result_based_confidence_threshold = params.postprocessing.result_based_confidence_threshold

        logger.info('Computing F-measure' + (' with auto threshold adjustment' if result_based_confidence_threshold else ''))
        f_measure_metrics = MetricsHelper.compute_f_measure(resultset,
                                                            result_based_confidence_threshold,
                                                            False,
                                                            False)

        if resultset.purpose is ResultsetPurpose.EVALUATION:
            # only set configurable params based on validation result set
            if result_based_confidence_threshold:
                best_confidence_threshold = f_measure_metrics.best_confidence_threshold.value
                if best_confidence_threshold is not None:
                    logger.info(f"Setting confidence_threshold to " f"{best_confidence_threshold} based on results")
                    # params.postprocessing.confidence_threshold = best_confidence_threshold
                else:
                    raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")

            # self.task_environment.set_configurable_parameters(params)
        logger.info(f"F-measure after evaluation: {f_measure_metrics.f_measure.value}")
        return f_measure_metrics.get_performance()


    def train(self, dataset: Dataset, output_model: Model, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        if self.train_model is None:
            raise ValueError("Training model is not initialized. Please load the trainable model to the task before training.")

        # Create new train_model if training from scratch.
        old_train_model = copy.deepcopy(self.train_model)
        if train_parameters is not None and train_parameters.train_on_empty_model:
            logger.info("Training from scratch, creating new model")
            # FIXME. Isn't it an overkill? Consider calling init_weights instead.
            self.train_model = self._create_model(config=self.config, from_scratch=True)

        # Evaluate model performance before training.
        logger.warning('PREEVALUATION')
        initial_performance = self._do_evaluation(self.inference_model, dataset.get_subset(Subset.VALIDATION))

        # Check for stop signal between pre-eval and training. If training is cancelled at this point,
        # old_train_model should be restored.
        if self.should_stop:
            self.should_stop = False
            logger.info('Training cancelled.')
            self.train_model = old_train_model
            return self.task_environment.model

        # Run training.
        self.time_monitor = TimeMonitorCallback(0, 0, 0, 0)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(self.config, dataset.get_subset(Subset.TRAINING),
            dataset.get_subset(Subset.VALIDATION), self.time_monitor, learning_curves)
        mm_train_dataset = build_dataset(training_config.data.train)
        self.is_training = True
        start_train_time = time.time()
        train_detector(model=self.train_model, dataset=mm_train_dataset, cfg=training_config, validate=True)
        training_duration = time.time() - start_train_time

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self.should_stop:
            self.should_stop = False
            logger.info('Training cancelled.')
            self.train_model = old_train_model
            return self.task_environment.model

        # Load the best weights and check if model has improved
        training_metrics = self._generate_training_metrics_group(learning_curves)
        best_checkpoint_path = os.path.join(training_config.work_dir, 'latest.pth')
        best_checkpoint = torch.load(best_checkpoint_path)
        # Create inference model as a copy of a train one.
        self.train_model.eval()
        self.train_model.load_state_dict(best_checkpoint['state_dict'])

        # Evaluate model performance after training.
        logger.warning('POSTEVALUATION')
        final_performance = self._do_evaluation(self.train_model, dataset.get_subset(Subset.VALIDATION))
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
            self.inference_model = copy.deepcopy(self.train_model)
            self._save_model(self.inference_model, output_model)
            output_model.performance = performance
            # output_model.tags = tags
            output_model.model_status = ModelStatus.SUCCESS
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
            # Restore old training model if training from scratch and not improved
            self.train_model = old_train_model

        self.is_training = False
        self.time_monitor = None
        return self.task_environment.model


    def _do_evaluation(self, model: torch.nn.Module, dataset: Dataset) -> float:
        """
        Performs evaluation of model using internal mAP metric.

        :return pretraining_performance (float): The performance score of the model
        """
        if model is not None:
            logger.info("Evaluating model.")
            _, pretraining_performance = self._do_inference(model, dataset, True)
        else:
            pretraining_performance = 0.0
        logger.info(f"Model performance: mAP = {pretraining_performance}")
        return pretraining_performance


    def _save_model(self, model: torch.nn.Module, output_model: Model):
        buffer = io.BytesIO()
        hyperparams = self.task_environment.get_configurable_parameters(ObjectDetectionConfig)
        config = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self.labels}
        modelinfo = {'model': model.state_dict(), 'config': config, 'mmdet_config': config_to_string(self.config),
                     'labels': labels, 'VERSION': 1}
        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())


    # def _persist_inference_model(self, dataset: Dataset, performance: Performance, training_duration: float):
    #     model_data = self._get_model_bytes(self.inference_model, self.inference_model.cfg)
    #     model = Model(project=self.task_environment.project,
    #                   model_storage=self.task_environment.task_node.model_storage,
    #                   task_node_id=self.task_environment.task_node.id,
    #                   configuration=self.task_environment.get_model_configuration(),
    #                   data_source_dict={"weights.pth": model_data},
    #                   tags=None,
    #                   performance=performance,
    #                   train_dataset=dataset,
    #                   training_duration=training_duration)
    #     self.task_environment.model = model


    def get_training_progress(self) -> float:
        """
        Calculate the progress of the current training

        :return: training progress in percent
        """
        if self.time_monitor is not None:
            return self.time_monitor.get_progress()
        return -1.0


    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self.should_stop = True
        stop_training_filepath = os.path.join(self.config.work_dir, '.stop_training')
        open(stop_training_filepath, 'a').close()


    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self.model_name)
        visualization_info_architecture = VisualizationInfo(name="Model architecture",
                                                            visualisation_type=VisualizationType.TEXT)
        output.append(MetricsGroup(metrics=[architecture],
                                   visualization_info=visualization_info_architecture))

        # Learning curves
        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            visualization_info = LineChartInfo(name=key, x_axis_label="Epoch", y_axis_label=key)
            output.append(MetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output


    @staticmethod
    def _get_model_from_bytes(blob: bytes) -> dict:
        buffer = io.BytesIO(blob)
        return torch.load(buffer)


    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> ObjectDetectionConfig:
        """
        Returns the configurable parameters.

        :param task_environment: Current task environment
        :return: Instance of ObjectDetectionConfig
        """
        return task_environment.get_configurable_parameters(ObjectDetectionConfig)


    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        """
        Called when the user changes the configurable parameters in the UI.

        :param task_environment: New task environment with updated configurable parameters
        """
        new_conf_params = self.get_configurable_parameters(task_environment)
        self.task_environment = task_environment
        set_hyperparams(self.config, new_conf_params)


    def _get_confidence(self, is_evaluation: bool) -> Tuple[float, float, bool]:
        """
        Retrieves the thresholds for confidence from the configurable parameters. If
        is_evaluation is True, the confidence threshold is set to 0 in order to compute optimum values
        for the thresholds. Also returns whether or not to perform nms across objects of different classes.

        :param is_evaluation: bool, True in case analysis is requested for evaluation

        :return confidence_threshold: float, threshold for prediction confidence
        """
        # conf_params = self.get_configurable_parameters(self.task_environment)
        conf_params = self.task_environment.get_configurable_parameters(ObjectDetectionConfig)
        confidence_threshold = conf_params.postprocessing.confidence_threshold
        result_based_confidence_threshold = conf_params.postprocessing.result_based_confidence_threshold
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


    def export(self,
               export_type: ExportType,
               output_model: OptimizedModel):
        optimized_model_precision = ModelPrecision.FP16
        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "export")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings("ignore", category=TracerWarning)
                export_model(self.inference_model, tempdir, target='openvino', precision=optimized_model_precision.name)
                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                output_model.set_data("openvino.bin", open(os.path.join(tempdir, bin_file), "rb").read())
                output_model.set_data("openvino.xml", open(os.path.join(tempdir, xml_file), "rb").read())
            except Exception as ex:
                raise RuntimeError("Optimization was unsuccessful.") from ex


    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space, ignore_errors=False)
