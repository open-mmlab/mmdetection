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
import torch
import json
import warnings
from collections import defaultdict
from typing import Optional, List, Tuple, Dict

import numpy as np

from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import (CurveMetric,
                                      LineChartInfo,
                                      MetricsGroup,
                                      Performance,
                                      ScoreMetric,
                                      InfoMetric,
                                      VisualizationType,
                                      VisualizationInfo)
from ote_sdk.entities.shapes.polygon import Rectangle
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.label import ScoredLabel

from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from sc_sdk.entities.annotation import Annotation
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.model import Model, ModelPrecision
from ote_sdk.entities.task_environment import TaskEnvironment


from sc_sdk.entities.model import Model, ModelStatus, NullModel

from sc_sdk.entities.resultset import ResultSet, ResultSetEntity
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask, OptimizationType
from ote_sdk.usecases.tasks.interfaces.export_interface import IExportTask, ExportType
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from sc_sdk.logging import logger_factory

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import Config
from mmdet.apis import train_detector, single_gpu_test, export_model
from mmdet.apis.ote.apis.detection.config_utils import patch_config
from mmdet.apis.ote.apis.detection.config_utils import set_hyperparams
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_training
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_testing
from mmdet.apis.ote.apis.detection.configuration import OTEDetectionConfig
from mmdet.apis.ote.apis.detection.configuration_enums import NNCFCompressionPreset

from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmdet.apis.train import create_nncf_model
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.integration.nncf import check_nncf_is_enabled
from mmdet.integration.nncf import wrap_nncf_model
from mmdet.integration.nncf.config import compose_nncf_config
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU
from mmdet.integration.nncf import is_state_nncf


logger = logger_factory.get_logger("NNCFDetectionTask")


COMPRESSION_MAP = {
    NNCFCompressionPreset.QUANTIZATION: "nncf_quantization",
    NNCFCompressionPreset.PRUNING: "nncf_pruning",
    NNCFCompressionPreset.QUANTIZATION_PRUNING: "nncf_pruning_quantization"
}


class NNCFDetectionTask(IOptimizationTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using OTEDetection.

        """
        logger.info(f"Loading NNCFDetectionTask.")
        self.scratch_space = tempfile.mkdtemp(prefix="ote-det-nncf-scratch-")
        logger.info(f"Scratch space created at {self.scratch_space}")

        self.task_environment = task_environment
        self.hyperparams = hyperparams = task_environment.get_hyper_parameters(OTEDetectionConfig)

        self.model_name = hyperparams.algo_backend.model_name
        self.labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmdet config.
        base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(base_dir, hyperparams.algo_backend.model)
        self.config = Config.fromfile(config_file_path)
        patch_config(self.config, self.scratch_space, self.labels, random_seed=42)
        set_hyperparams(self.config, hyperparams)

        # NNCF part
        self.compression_ctrl = None
        nncf_config_path = os.path.join(base_dir, hyperparams.nncf_optimization.config)

        with open(nncf_config_path) as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        optimization_type = COMPRESSION_MAP[hyperparams.nncf_optimization.preset]

        optimization_config = compose_nncf_config(common_nncf_config, [optimization_type])
        self.config.update(optimization_config)

        # Create and initialize PyTorch model.
        check_nncf_is_enabled()
        self.compression_ctrl, self.model = self._load_model(task_environment.model)

        # Extra control variables.
        self.training_work_dir = None
        self.is_training = False
        self.should_stop = False
        self.time_monitor = None

    @staticmethod
    def extract_model_and_compression_states(resuming_checkpoint: Optional[Dict] = None):
        if resuming_checkpoint is None:
            return None, None
        model_state_dict = resuming_checkpoint.get("model" if "model" in resuming_checkpoint else "state_dict")
        compression_state = resuming_checkpoint.get("compression_state")
        return model_state_dict, compression_state

    def _load_model(self, model: Model):
        if model != NullModel():
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self.config, from_scratch=True)
            compression_ctrl = None
            try:
                if is_state_nncf(model_data):
                    from mmdet.apis.fake_input import get_fake_input
                    compression_ctrl, model = wrap_nncf_model(
                        model,
                        self.config,
                        init_state_dict=model_data,
                        get_fake_input_func=get_fake_input
                    )
                else:
                    # TODO: was only model, state_dict was depricated
                    if 'model' in model_data:
                        model.load_state_dict(model_data['model'])
                    elif 'state_dict' in model_data:
                        model.load_state_dict(model_data['state_dict'])
                    else:
                        raise ValueError("Could not load the saved model. No model ot state_dict key.")
                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self.model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self.config, from_scratch=False)
            logger.info(f"No trained model in project yet. Created new model with '{self.model_name}' "
                        f"architecture and general-purpose pretrained weights.")
        return compression_ctrl, model

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
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

    def infer(self, dataset: Dataset, inference_parameters: Optional[InferenceParameters] = None) -> Dataset:
        """ Analyzes a dataset using the latest inference model. """

        is_evaluation = inference_parameters is not None and inference_parameters.is_evaluation
        confidence_threshold = self._get_confidence_threshold(is_evaluation)
        logger.info(f'Confidence threshold {confidence_threshold}')

        prediction_results, _ = self._infer_detector(self.model, self.config, dataset, False)

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

                    assigned_label = [ScoredLabel(self.labels[label_idx],
                                                  probability=probability)]
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    shapes.append(Annotation(
                        Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        labels=assigned_label))

            dataset_item.append_annotations(shapes)

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
                 output_result_set: ResultSet,
                 evaluation_metric: Optional[str] = None):
        """ Computes performance on a resultset """
        params = self.hyperparams

        result_based_confidence_threshold = params.postprocessing.result_based_confidence_threshold

        logger.info('Computing F-measure' + (' with auto threshold adjustment' if result_based_confidence_threshold else ''))
        f_measure_metrics = MetricsHelper.compute_f_measure(output_result_set,
                                                            result_based_confidence_threshold,
                                                            False,
                                                            False)

        if output_result_set.purpose is ResultSetEntity.EVALUATION:
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

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: Dataset,
        output_model: Model,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self.config

        self.time_monitor = TimeMonitorCallback(0, 0, 0, 0, update_progress_callback=lambda _: None)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, self.time_monitor, learning_curves)
        mm_train_dataset = build_dataset(training_config.data.train)

        if not self.compression_ctrl:
            self.compression_ctrl, self.model = create_nncf_model(
                self.model,
                mm_train_dataset,
                training_config,
                False)

        # Run training.
        self.training_work_dir = training_config.work_dir
        self.is_training = True
        self.model.train()

        train_detector(model=self.model,
                       dataset=mm_train_dataset,
                       cfg=training_config,
                       validate=True,
                       compression_ctrl=self.compression_ctrl)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled
        if self.should_stop:
            logger.info('Training cancelled.')
            self.should_stop = False
            self.is_training = False
            self.time_monitor = None
            return

        output_model.model_status = ModelStatus.SUCCESS

        self.is_training = False
        self.time_monitor = None

    def save_model(self, output_model: Model):
        buffer = io.BytesIO()
        hyperparams = self.task_environment.get_hyper_parameters(OTEDetectionConfig)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self.labels}
        modelinfo = {
            'compression_state': self.compression_ctrl.get_compression_state(),
            'meta': { # To be comparable with old mmdetection
                'config': {
                    'nncf_config': self.config["nncf_config"],
                },
                'nncf_enable_compression': True,
            },
            'model': self.model.state_dict(),
            'config': hyperparams_str,
            'labels': labels,
            'VERSION': 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())


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
        stop_training_filepath = os.path.join(self.training_work_dir, '.stop_training')
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


    def _get_confidence_threshold(self, is_evaluation: bool) -> float:
        """
        Retrieves the threshold for confidence from the configurable parameters. If
        is_evaluation is True, the confidence threshold is set to 0 in order to compute optimum values
        for the thresholds.

        :param is_evaluation: bool, True in case analysis is requested for evaluation

        :return confidence_threshold: float, threshold for prediction confidence
        """

        hyperparams = self.hyperparams
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
               output_model: Model):
        assert export_type == ExportType.OPENVINO
        optimized_model_precision = ModelPrecision.FP32
        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "export")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings("ignore", category=TracerWarning)
                if torch.cuda.is_available():
                    model = self.model.cuda(self.config.gpu_ids[0])
                else:
                    model = self.model.cpu()
                export_model(model, self.config, tempdir,
                             target='openvino', precision=optimized_model_precision.name)
                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                with open(os.path.join(tempdir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(tempdir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                output_model.precision = [optimized_model_precision]
            except Exception as ex:
                raise RuntimeError("Optimization was unsuccessful.") from ex


    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space, ignore_errors=False)
