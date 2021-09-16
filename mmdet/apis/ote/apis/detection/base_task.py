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
import logging
import numpy as np
import os
import shutil
import tempfile
import torch
import warnings
from typing import List, Optional, Tuple

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.label import ScoredLabel
from ote_sdk.entities.metrics import (CurveMetric, InfoMetric, LineChartInfo,
                                      MetricsGroup, VisualizationInfo,
                                      VisualizationType)
from ote_sdk.entities.model import ModelStatus, ModelPrecision, ModelEntity, ModelFormat, ModelOptimizationType
from ote_sdk.entities.resultset import ResultSetEntity, ResultsetPurpose
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import default_progress_callback
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType
from ote_sdk.usecases.tasks.interfaces.export_interface import IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from sc_sdk.entities.annotation import Annotation
from sc_sdk.entities.datasets import Dataset

from mmdet.apis import export_model, single_gpu_test
from mmdet.apis.ote.apis.detection.config_utils import patch_config
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_testing
from mmdet.apis.ote.apis.detection.config_utils import set_hyperparams
from mmdet.apis.ote.apis.detection.configuration import OTEDetectionConfig
from mmdet.apis.ote.apis.detection.ote_utils import InferenceProgressCallback

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU


logger = logging.getLogger(__name__)


class OTEBaseTask(IInferenceTask, IExportTask, IEvaluationTask, IUnload):

    _task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        self._task_environment = task_environment

        logger.info(f"Loading OTEDetectionTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-det-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._hyperparams = hyperparams = task_environment.get_hyper_parameters(OTEDetectionConfig)

        self._model_name = hyperparams.algo_backend.model_name
        self._labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmdet config.
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(self._base_dir, hyperparams.algo_backend.model)
        self._config = Config.fromfile(config_file_path)
        patch_config(self._config, self._scratch_space, self._labels, random_seed=42)
        set_hyperparams(self._config, hyperparams)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
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


    def infer(self, dataset: Dataset, inference_parameters: Optional[InferenceParameters] = None) -> Dataset:
        """ Analyzes a dataset using the latest inference model. """
        set_hyperparams(self._config, self._hyperparams)

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            is_evaluation = inference_parameters.is_evaluation
        else:
            is_evaluation = False
            update_progress_callback = default_progress_callback

        time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        def pre_hook(module, input):
            time_monitor.on_test_batch_begin(None, None)

        def hook(module, input, output):
            time_monitor.on_test_batch_end(None, None)

        pre_hook_handle = self._model.register_forward_pre_hook(pre_hook)
        hook_handle = self._model.register_forward_hook(hook)

        confidence_threshold = self._get_confidence_threshold(is_evaluation)
        logger.info(f'Confidence threshold {confidence_threshold}')

        prediction_results, _ = self._infer_detector(self._model, self._config, dataset, False)

        # Loop over dataset again to assign predictions. Convert from MMDetection format to OTE format
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

                    assigned_label = [ScoredLabel(self._labels[label_idx],
                                                  probability=probability)]
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    shapes.append(Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label))

            dataset_item.append_annotations(shapes)

        pre_hook_handle.remove()
        hook_handle.remove()

        return dataset


    @staticmethod
    def _infer_detector(model: torch.nn.Module, config: Config, dataset: Dataset,
                        eval: Optional[bool] = False, metric_name: Optional[str] = 'mAP') -> Tuple[List, float]:
        model.eval()
        test_config = prepare_for_testing(config, dataset)
        mm_val_dataset = build_dataset(test_config.data.test)
        batch_size = 1
        mm_val_dataloader = build_dataloader(mm_val_dataset,
                                             samples_per_gpu=batch_size,
                                             workers_per_gpu=test_config.data.workers_per_gpu,
                                             num_gpus=1,
                                             dist=False,
                                             shuffle=False)
        if torch.cuda.is_available():
            eval_model = MMDataParallel(model.cuda(test_config.gpu_ids[0]),
                                        device_ids=test_config.gpu_ids)
        else:
            eval_model = MMDataCPU(model)
        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        eval_predictions = single_gpu_test(eval_model, mm_val_dataloader, show=False)

        metric = None
        if eval:
            metric = mm_val_dataset.evaluate(eval_predictions, metric=metric_name)[metric_name]
        return eval_predictions, metric


    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        """ Computes performance on a resultset """
        params = self._hyperparams

        result_based_confidence_threshold = params.postprocessing.result_based_confidence_threshold

        logger.info('Computing F-measure' + (' with auto threshold adjustment' if result_based_confidence_threshold else ''))
        f_measure_metrics = MetricsHelper.compute_f_measure(output_result_set,
                                                            result_based_confidence_threshold,
                                                            False,
                                                            False)

        if output_result_set.purpose is ResultsetPurpose.EVALUATION:
            # only set configurable params based on validation result set
            if result_based_confidence_threshold:
                best_confidence_threshold = f_measure_metrics.best_confidence_threshold.value
                if best_confidence_threshold is not None:
                    logger.info(f"Setting confidence_threshold to " f"{best_confidence_threshold} based on results")
                    # params.postprocessing.confidence_threshold = best_confidence_threshold
                else:
                    raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")

            # self._task_environment.set_configurable_parameters(params)
        logger.info(f"F-measure after evaluation: {f_measure_metrics.f_measure.value}")

        output_result_set.performance = f_measure_metrics.get_performance()


    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self._model_name)
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


    def _get_confidence_threshold(self, is_evaluation: bool) -> float:
        """
        Retrieves the threshold for confidence from the configurable parameters. If
        is_evaluation is True, the confidence threshold is set to 0 in order to compute optimum values
        for the thresholds.

        :param is_evaluation: bool, True in case analysis is requested for evaluation

        :return confidence_threshold: float, threshold for prediction confidence
        """

        hyperparams = self._hyperparams
        confidence_threshold = hyperparams.postprocessing.confidence_threshold
        result_based_confidence_threshold = hyperparams.postprocessing.result_based_confidence_threshold
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
               output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO
        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "export")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings("ignore", category=TracerWarning)
                if torch.cuda.is_available():
                    model = self._model.cuda(self._config.gpu_ids[0])
                else:
                    model = self._model.cpu()
                export_model(model, self._config, tempdir,
                             target='openvino', precision=optimized_model_precision.name)
                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                with open(os.path.join(tempdir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(tempdir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                output_model.precision = [optimized_model_precision]
                output_model.model_status = ModelStatus.SUCCESS
            except Exception as ex:
                output_model.model_status = ModelStatus.FAILED
                raise RuntimeError("Optimization was unsuccessful.") from ex


    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)