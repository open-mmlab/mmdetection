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
import os
from collections import defaultdict
from typing import List, Optional

import torch
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.metrics import (CurveMetric, InfoMetric, LineChartInfo, MetricsGroup, Performance, ScoreMetric,
                                      VisualizationInfo, VisualizationType)
from ote_sdk.entities.model import ModelEntity, ModelStatus
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.train_parameters import TrainParameters, default_progress_callback
from ote_sdk.usecases.tasks.interfaces.training_interface import ITrainingTask

from mmdet.apis import train_detector
from mmdet.apis.ote.apis.detection.config_utils import prepare_for_training, set_hyperparams
from mmdet.apis.ote.apis.detection.inference_task import OTEDetectionInferenceTask
from mmdet.apis.ote.apis.detection.ote_utils import TrainingProgressCallback
from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmdet.datasets import build_dataset

logger = logging.getLogger(__name__)


class OTEDetectionTrainingTask(OTEDetectionInferenceTask, ITrainingTask):

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


    def train(self, dataset: DatasetEntity, output_model: ModelEntity, train_parameters: Optional[TrainParameters] = None):
        """ Trains a model on a dataset """

        set_hyperparams(self._config, self._hyperparams)

        train_dataset = dataset.get_subset(Subset.TRAINING)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        config = self._config

        # Create new model if training from scratch.
        old_model = copy.deepcopy(self._model)

        # Evaluate model performance before training.
        _, initial_performance = self._infer_detector(self._model, config, val_dataset, True)
        logger.info(f'initial_performance = {initial_performance}')

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
        update_progress_callback = default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
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
