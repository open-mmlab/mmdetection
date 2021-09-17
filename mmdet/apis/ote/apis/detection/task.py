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
import logging
import os
from collections import defaultdict
from typing import Optional

import torch
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.metrics import Performance, ScoreMetric
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask
from sc_sdk.entities.datasets import Dataset

from mmdet.apis import train_detector
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_training, set_hyperparams
from mmdet.apis.ote.apis.detection.configuration import OTEDetectionConfig
from mmdet.apis.ote.apis.detection.ote_utils import InferenceProgressCallback, TrainingProgressCallback
from mmdet.apis.ote.apis.detection.base_task import OTEBaseTask
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook

from mmdet.datasets import build_dataset


logger = logging.getLogger(__name__)


class OTEDetectionTask(OTEBaseTask, ITrainingTask):

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using OTEDetection.
        """
        super().__init__(task_environment)

        # Create and initialize PyTorch model.
        self._model = self._load_model(task_environment.model)


    def _load_model(self, model: ModelEntity):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            model = self._create_model(self._config, from_scratch=True)

            try:
                model.load_state_dict(model_data['model'])
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
        return model


    def train(self, dataset: Dataset, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        set_hyperparams(self._config, self._hyperparams)

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        # Create new model if training from scratch.
        old_model = copy.deepcopy(self._model)

        # Evaluate model performance before training.
        _, initial_performance = self._infer_detector(self._model, config, val_dataset, True)

        # Check for stop signal between pre-eval and training. If training is cancelled at this point,
        # old_model should be restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            self._training_work_dir = None
            return

        # Run training.
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback
        time_monitor = TrainingProgressCallback(update_progress_callback)
        learning_curves = defaultdict(OTELoggerHook.Curve)
        training_config = prepare_for_training(config, train_dataset, val_dataset, time_monitor, learning_curves)
        self._training_work_dir = training_config.work_dir
        mm_train_dataset = build_dataset(training_config.data.train)
        self._is_training = True
        self._model.train()
        train_detector(model=self._model, dataset=mm_train_dataset, cfg=training_config, validate=True)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self._should_stop:
            logger.info('Training cancelled.')
            self._model = old_model
            self._should_stop = False
            self._is_training = False
            return

        # Load the best weights and check if model has improved.
        training_metrics = self._generate_training_metrics_group(learning_curves)
        best_checkpoint_path = os.path.join(training_config.work_dir, 'latest.pth')
        best_checkpoint = torch.load(best_checkpoint_path)
        self._model.load_state_dict(best_checkpoint['state_dict'])

        # Evaluate model performance after training.
        _, final_performance = self._infer_detector(self._model, config, val_dataset, True)
        improved = final_performance > initial_performance

        # Return a new model if model has improved, or there is no model yet.
        if improved or self._task_environment.model is None:
            if improved:
                logger.info("Training finished, and it has an improved model")
            else:
                logger.info("First training round, saving the model.")
            # Add mAP metric and loss curves
            performance = Performance(score=ScoreMetric(value=final_performance, name="mAP"),
                                      dashboard_metrics=training_metrics)
            logger.info('FINAL MODEL PERFORMANCE\n' + str(performance))
            self.save_model(output_model)
            output_model.performance = performance
            output_model.model_status = ModelStatus.SUCCESS
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
            # Restore old training model if training from scratch and not improved
            self._model = old_model

        self._is_training = False


    def save_model(self, output_model: ModelEntity):
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEDetectionConfig)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        modelinfo = {'model': self._model.state_dict(), 'config': hyperparams_str, 'labels': labels, 'VERSION': 1}
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
