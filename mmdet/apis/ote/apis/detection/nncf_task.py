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
from mmdet.apis.ote.apis.detection.base_task import OTEBaseTask

logger = logger_factory.get_logger("NNCFDetectionTask")


COMPRESSION_MAP = {
    NNCFCompressionPreset.QUANTIZATION: "nncf_quantization",
    NNCFCompressionPreset.PRUNING: "nncf_pruning",
    NNCFCompressionPreset.QUANTIZATION_PRUNING: "nncf_pruning_quantization"
}


class NNCFDetectionTask(OTEBaseTask, IOptimizationTask):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using OTEDetection.

        """
        super().__init__(task_environment)
        logger.info(f"Loading OTEDetectionTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-det-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment
        self._hyperparams = hyperparams = task_environment.get_hyper_parameters(OTEDetectionConfig)

        self._model_name = hyperparams.algo_backend.model_name
        self._labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmdet config.
        base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(base_dir, hyperparams.algo_backend.model)
        self._config = Config.fromfile(config_file_path)
        patch_config(self._config, self._scratch_space, self._labels, random_seed=42)
        set_hyperparams(self._config, hyperparams)

        # NNCF part
        self._compression_ctrl = None
        nncf_config_path = os.path.join(base_dir, hyperparams.nncf_optimization.config)

        with open(nncf_config_path) as nncf_config_file:
            common_nncf_config = json.load(nncf_config_file)

        optimization_type = COMPRESSION_MAP[hyperparams.nncf_optimization.preset]

        optimization_config = compose_nncf_config(common_nncf_config, [optimization_type])
        self._config.update(optimization_config)

        # Create and initialize PyTorch model.
        check_nncf_is_enabled()
        self._compression_ctrl, self._model = self._load_model(task_environment.model)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False

    @staticmethod
    def extract_model_and_compression_states(resuming_checkpoint: Optional[Dict] = None):
        if resuming_checkpoint is None:
            return None, None
        # TODO remove state_dict
        model_state_dict = resuming_checkpoint.get("model" if "model" in resuming_checkpoint else "state_dict")
        compression_state = resuming_checkpoint.get("compression_state")
        return model_state_dict, compression_state

    def _load_model(self, model: Model):
        if model != NullModel():
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._config, from_scratch=True)
            compression_ctrl = None
            try:
                if is_state_nncf(model_data):
                    from mmdet.apis.fake_input import get_fake_input
                    compression_ctrl, model = wrap_nncf_model(
                        model,
                        self._config,
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
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._config, from_scratch=False)
            logger.info(f"No trained model in project yet. Created new model with '{self._model_name}' "
                        f"architecture and general-purpose pretrained weights.")
        return compression_ctrl, model

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
        config = self._config

        time_monitor = TimeMonitorCallback(0, 0, 0, 0, update_progress_callback=lambda _: None)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        mm_train_dataset = build_dataset(training_config.data.train)

        if not self._compression_ctrl:
            self._compression_ctrl, self._model = create_nncf_model(
                self._model,
                mm_train_dataset,
                training_config,
                False)

        # Run training.
        self.training_work_dir = training_config.work_dir
        self.is_training = True
        self._model.train()

        train_detector(model=self._model,
                       dataset=mm_train_dataset,
                       cfg=training_config,
                       validate=True,
                       compression_ctrl=self._compression_ctrl)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled
        if self._should_stop:
            logger.info('Training cancelled.')
            self._should_stop = False
            self._is_training = False
            return

        output_model.model_status = ModelStatus.SUCCESS

        self._is_training = False

    def save_model(self, output_model: Model):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEDetectionConfig)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {
            'compression_state': self._compression_ctrl.get_compression_state(),
            'meta': {
                'config': self._config,
                'nncf_enable_compression': True,
            },
            'model': self._model.state_dict(),
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
