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

import io
import json
import logging
import os
import torch
from collections import defaultdict
from typing import Optional

from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.model import ModelEntity
from ote_sdk.entities.model import ModelStatus
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from ote_sdk.usecases.tasks.interfaces.optimization_interface import IOptimizationTask
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType

from sc_sdk.entities.datasets import Dataset

from mmcv.utils import Config
from mmdet.apis import train_detector
from mmdet.apis.ote.apis.detection.config_utils import patch_config
from mmdet.apis.ote.apis.detection.config_utils import set_hyperparams
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_training
from mmdet.apis.ote.apis.detection.configuration import OTEDetectionConfig
from mmdet.apis.ote.apis.detection.configuration_enums import NNCFCompressionPreset

from mmdet.apis.fake_input import get_fake_input
from mmdet.apis.ote.apis.detection.base_task import OTEBaseTask
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmdet.apis.ote.apis.detection.ote_utils import TrainingProgressCallback
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmdet.integration.nncf import check_nncf_is_enabled
from mmdet.integration.nncf import is_state_nncf
from mmdet.integration.nncf import wrap_nncf_model
from mmdet.integration.nncf.config import compose_nncf_config


logger = logging.getLogger(__name__)


COMPRESSION_MAP = {
    NNCFCompressionPreset.QUANTIZATION: "nncf_quantization",
    NNCFCompressionPreset.PRUNING: "nncf_pruning",
    NNCFCompressionPreset.QUANTIZATION_PRUNING: "nncf_pruning_quantization"
}


class NNCFDetectionTask(OTEBaseTask, IOptimizationTask):

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for compressing object detection models using NNCF.
        """
        super().__init__(task_environment)

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
        nncf_config_path = os.path.join(base_dir, "compression_config.json")

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

    def _load_model(self, model: ModelEntity):
        compression_ctrl = None
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._config, from_scratch=True)
            try:
                if is_state_nncf(model_data):
                    compression_ctrl, model = wrap_nncf_model(
                        model,
                        self._config,
                        init_state_dict=model_data,
                        get_fake_input_func=get_fake_input
                    )
                    logger.info("Loaded model weights from Task Environment and wrapped by NNCF")
                else:
                    try:
                        model.load_state_dict(model_data['model'])
                        logger.info(f"Loaded model weights from Task Environment")
                        logger.info(f"Model architecture: {self._model_name}")
                    except BaseException as ex:
                        raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                            from ex

                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            raise ValueError(f"No trained model in project. NNCF require pretrained weights to compress the model")
        return compression_ctrl, model

    def _create_compressed_model(self, dataset, config):
        init_dataloader = build_dataloader(
            dataset,
            config.data.samples_per_gpu,
            config.data.workers_per_gpu,
            len(config.gpu_ids),
            dist=False,
            seed=config.seed)

        self._compression_ctrl, self._model = wrap_nncf_model(
            self._model,
            config,
            init_dataloader,
            get_fake_input)

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: Dataset,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters],
    ):
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        time_monitor = TrainingProgressCallback(update_progress_callback)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        mm_train_dataset = build_dataset(training_config.data.train)

        if torch.cuda.is_available():
            self._model.cuda(training_config.gpu_ids[0])

        # Initialize NNCF parts if start from not compressed model
        if not self._compression_ctrl:
            self._create_compressed_model(mm_train_dataset, training_config)

        # Run training.
        self._training_work_dir = training_config.work_dir
        self._is_training = True
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

        self.save_model(output_model)

        output_model.model_status = ModelStatus.SUCCESS
        self._is_training = False

    def save_model(self, output_model: ModelEntity):
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

    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        logger.info("Cancel training requested.")
        self._should_stop = True
        stop_training_filepath = os.path.join(self._training_work_dir, '.stop_training')
        open(stop_training_filepath, 'a').close()
