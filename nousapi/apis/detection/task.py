import os
import torch
import tempfile
import copy
import io
import glob
import shutil
import json
import time
import mmcv

from itertools import compress
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from noussdk.entities.analyse_parameters import AnalyseParameters
from noussdk.entities.datasets import Dataset
from noussdk.entities.metrics import CurveMetric, LineChartInfo, MetricsGroup, Performance, ScoreMetric, InfoMetric, \
    VisualizationType, VisualizationInfo
from noussdk.entities.optimized_model import OptimizedModel, OpenVINOModel, Precision
from noussdk.entities.task_environment import TaskEnvironment
from noussdk.entities.train_parameters import TrainParameters
from noussdk.entities.label_relations import ScoredLabel
from noussdk.entities.model import Model, NullModel
from noussdk.entities.shapes.box import Box
from noussdk.entities.resultset import ResultSetEntity, ResultsetPurpose

from noussdk.usecases.evaluation.basic_operations import get_nms_filter
from noussdk.usecases.evaluation.metrics_helper import MetricsHelper
from noussdk.usecases.tasks.image_deep_learning_task import ImageDeepLearningTask
from noussdk.usecases.tasks.interfaces.configurable_parameters_interface import IConfigurableParameters
from noussdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
from noussdk.usecases.tasks.interfaces.unload_interface import IUnload

from noussdk.logging import logger_factory

from noussdk.utils.openvino_tools import generate_openvino_model

from mmdet.apis import train_detector, get_root_logger, set_random_seed, single_gpu_test, inference_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.core import preprocess_example_input, generate_inputs_and_wrap_model

from mmcv.parallel import MMDataParallel
from mmcv.utils import Config
from mmcv.runner import save_checkpoint

from tasks.mmdetection_tasks.detection import MMDetectionParameters
from tasks.mmdetection_tasks.config import MMDetectionConfigManager, MMDetectionTaskType

# The following imports are needed to register the custom datasets and hooks for NOUS as modules in the
# mmdetection framework. They are not used directly in this file, but they have to be here for the registration to work
from tasks.mmdetection_tasks.datasets import NOUSDataset
from tasks.mmdetection_tasks.utils import CancelTrainingHook, FixedMomentumUpdaterHook, LoadImageFromNOUSDataset, \
    EpochRunnerWithCancel, LoadAnnotationFromNOUSDataset

logger = logger_factory.get_logger("MMDetectionTask")


def safe_inference_detector(model: torch.nn.Module, image: np.ndarray) -> List[np.array]:
    """
    Wrapper function to perform inference without breaking the model config.
    The mmdetection function 'inference_detector' modifies model.cfg in place, causing subsequent evaluation calls to
    single_gpu_test to break.
    To avoid this, we make a copy of the config and restore after inference.

    :param model: model to use for inference
    :param image: image to infer
    :return results: list of detection results
    """
    model_cfg = copy.deepcopy(model.cfg)
    output = inference_detector(model, image)
    model.cfg = model_cfg
    return output


class MMObjectDetectionTask(ImageDeepLearningTask, IConfigurableParameters, IModelOptimizer, IUnload):

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for training object detection models using the MMDetection framework.

        """
        logger.info(f"Loading MMDetection task of type 'Detection' with task ID {task_environment.task_node.id}.")

        # Temp directory to store logs and model checkpoints
        self.scratch_space = tempfile.mkdtemp(prefix="nous-scratch-")
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

        # Always use 3 channels for now
        self.in_channels = 3

        # n_samples is needed for progress estimation
        self.n_samples_in_current_training_set = 0

        # Model initialization.
        self.train_model = None
        self.inference_model = None
        self.load_model(self.task_environment)

    @staticmethod
    def create_model(config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: Model in training mode
        """
        model_cfg = copy.deepcopy(config.model)
        if from_scratch:
            model_cfg.pretrained = None
        return build_detector(model_cfg)

    def analyse(self, dataset: Dataset, analyse_parameters: Optional[AnalyseParameters] = None) -> Dataset:
        """ Analyzes a dataset using the latest inference model. """
        is_evaluation = analyse_parameters is not None and analyse_parameters.is_evaluation
        confidence_threshold, nms_threshold, cross_class_nms = self.get_confidence_and_nms_thresholds(is_evaluation)

        batch_size = self.config_manager.config.data.samples_per_gpu

        prediction_results = []
        if len(dataset) <= batch_size:
            # For small datasets,  just loop over the dataset_items and perform inference one by one
            for dataset_item in dataset:
                output = safe_inference_detector(self.inference_model, dataset_item.numpy)
                prediction_results.append(output)
        else:
            # For larger datasets, build a data_loader to perform the analysis. This is much faster than one by one
            # inference.
            # First, update the dataset in the model config. Regardless of NOUS DatasetPurpose, the dataset is always
            # set to the mmdetection test dataset
            # TODO: Running this with a dataloader greatly speeds up analysis, however it results in a warning:
            #   UserWarning: MongoClient opened before fork. Create MongoClient only after forking. See PyMongo's
            #   documentation for details: https://pymongo.readthedocs.io/en/stable/faq.html#is-pymongo-fork-safe
            #   Link to JIRA ticket:
            #   https://cosmonio.atlassian.net/browse/NI-660?atlOrigin=eyJpIjoiOGJmZjE4M2FmMjY1NDBmY2E0Yzk0N2NiYzk4ZTc5NjIiLCJwIjoiaiJ9
            self.inference_model.cfg.data.test.nous_dataset = dataset
            mm_test_dataset = build_dataset(copy.deepcopy(self.inference_model.cfg.data.test))
            # Use a single gpu for testing. Set in both mm_test_dataloader and prediction_model
            mm_test_dataloader = build_dataloader(mm_test_dataset, samples_per_gpu=batch_size, num_gpus=1, dist=False,
                                                  workers_per_gpu=self.config_manager.config.data.workers_per_gpu,
                                                  shuffle=False)
            prediction_model = MMDataParallel(self.inference_model.cuda(self.config_manager.config.gpu_ids[0]),
                                              device_ids=self.config_manager.config.gpu_ids)
            prediction_results = single_gpu_test(prediction_model, mm_test_dataloader, show=False)

        # Loop over dataset again to assign predictions. Convert from MMdetection format to NOUS format
        for ii, dataset_item in enumerate(dataset):
            output = prediction_results[ii]
            width = dataset_item.width
            height = dataset_item.height

            shapes = []
            for label_idx, detections in enumerate(output):
                for i in range(detections.shape[0]):
                    probability = float(detections[i, 4])

                    if probability < confidence_threshold:
                        continue

                    assigned_label = [ScoredLabel(self.config_manager.config.labels[label_idx],
                                                  probability=probability)]
                    if detections[i, 3] - detections[i, 1] <= 0 or detections[i, 2] - detections[i, 0] <= 0:
                        continue

                    shapes.append(Box(x1=float(detections[i, 0])/width,
                                      y1=float(detections[i, 1])/height,
                                      x2=float(detections[i, 2])/width,
                                      y2=float(detections[i, 3])/height,
                                      labels=assigned_label))

            if nms_threshold < 1.0:
                nms_filter = get_nms_filter(shapes, nms_threshold, cross_class_nms)
                shapes = list(compress(shapes, nms_filter))

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
            model_data_bytes = io.BytesIO(model.data)
            model_data = torch.load(model_data_bytes)
            model_config = self.config_manager.config_from_string(model_data['config'])
            torch_model = self.create_model(config=model_config, from_scratch=True)

            try:
                torch_model.load_state_dict(model_data['state_dict'])
                logger.info(f"Loaded model weights from: {model.data_url}")
                logger.info(f"Model architecture: {model_config.model.type}")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex

            self.inference_model = torch_model
            if model_config.model.type == self.config_manager.config.model.type:
                # If the model architecture hasn't changed in the configurable parameters, train model builds upon
                # the loaded inference model weights and we can just copy the inference model to train_model
                self.train_model = copy.deepcopy(self.inference_model)
            else:
                # If the model architecture has changed, then we start training a model with the desired architecture
                self.train_model = self.create_model(config=self.config_manager.config, from_scratch=False)
                logger.info(f"Model architecture has changed in configurable parameters. Initialized train model with "
                            f"{self.config_manager.config.model.type} architecture.")
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file. These are ImageNet pretrained
            model_config = self.config_manager.config_copy
            torch_model = self.create_model(config=model_config, from_scratch=False)
            self.train_model = torch_model
            self.inference_model = copy.deepcopy(self.train_model)
            logger.info(f"No trained model in project yet. Created new model with {model_config.model.type} "
                        f"architecture and ImageNet pretrained weights.")

        # Set the model configs. Inference always uses the config that came with the model, while training uses the
        # latest config in the config_manager
        self.inference_model.cfg = model_config
        self.train_model.cfg = self.config_manager.config_copy

        self.inference_model.eval()

    def _create_training_checkpoint_dirs(self) -> str:
        """
        Create directory to store checkpoints for next training run. Also sets experiment name and updates config

        :return train_round_checkpoint_dir: str, path to checkpoint dir
        """
        # Create new directory for checkpoints
        checkpoint_dirs = glob.glob(os.path.join(self.scratch_space, "checkpoints_round_*"))
        train_round_checkpoint_dir = os.path.join(self.scratch_space, f"checkpoints_round_{len(checkpoint_dirs)}")
        os.makedirs(train_round_checkpoint_dir)
        logger.info(f"Checkpoints and logs for this training run are stored in {train_round_checkpoint_dir}")
        self.config_manager.config.work_dir = train_round_checkpoint_dir
        self.config_manager.config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"

        # Save training config for debugging. It is saved in the checkpoint dir for this training round
        self.config_manager.save_config_to_file()

        return train_round_checkpoint_dir

    def _is_train_from_scratch(self, train_parameters: TrainParameters) -> Tuple[torch.nn.Module, bool]:
        """
        Checks whether to train a model from scratch.

        :param train_parameters: parameters with which training has been called.
        :return (old_train_model, train_from_scratch): old_train_model holds the old training model.
            train_from_scratch is True in case training from scratch, False otherwise
        """
        old_train_model = copy.deepcopy(self.train_model)
        if train_parameters is not None and train_parameters.train_on_empty_model:
            logger.info("Training from scratch, created new model")
            self.train_model = self.create_model(config=self.config_manager.config, from_scratch=True)
            train_from_scratch = True
        else:
            train_from_scratch = False
        return old_train_model, train_from_scratch

    def _do_pre_evaluation(self, dataset: Dataset) -> Tuple[float, bool]:
        """
        Performs evaluation of model before training.

        :return pretraining_performance, compare_pre_and_post_training_performance (float, bool): The performance score
            of the model before training, and whether or not to compare performance before and after training
        """
        # Pre-evaluation
        if self.inference_model is not None:
            logger.info("Pre-evaluating inference model.")
            # Build the dataset with the correct data configuration. Config has to come from the model, not the
            # config_manager, because architecture might have changed
            self.inference_model = self.config_manager.update_dataset_subsets(dataset=dataset,
                                                                              model=self.inference_model)
            mm_val_dataset = build_dataset(copy.deepcopy(self.inference_model.cfg.data.val))
            # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
            mm_val_dataloader = build_dataloader(mm_val_dataset,
                                                 samples_per_gpu=self.config_manager.config.data.samples_per_gpu,
                                                 workers_per_gpu=self.config_manager.config.data.workers_per_gpu,
                                                 num_gpus=1,
                                                 dist=False,
                                                 shuffle=False)
            eval_model = MMDataParallel(self.inference_model.cuda(self.config_manager.config.gpu_ids[0]),
                                        device_ids=self.config_manager.config.gpu_ids)
            pre_eval_predictions = single_gpu_test(eval_model, mm_val_dataloader, show=False)
            pre_eval_results = mm_val_dataset.evaluate(pre_eval_predictions, metric='mAP')
            pretraining_performance = pre_eval_results['mAP']
            logger.info(f"Pre-training model performance: mAP = {pretraining_performance}")
            compare_pre_and_post_training_performance = True
        else:
            compare_pre_and_post_training_performance = False
            pretraining_performance = 0.0
        return pretraining_performance, compare_pre_and_post_training_performance

    def _do_model_training(self, mm_train_dataset):
        """
        Trains the model.

        :param mm_train_dataset: training dataset in mmdetection format
        :return training_duration: Duration of the training round.
        """
        # Length of the training dataset is required for progress reporting, hence it is passed to the task class
        self.n_samples_in_current_training_set = len(mm_train_dataset)

        # Set model config to the most up to date version. Not 100% sure if this is even needed, setting just in case
        self.train_model.cfg = self.config_manager.config_copy

        # Train the model. Training modifies mmdet config in place, so make a deepcopy
        self.is_training = True
        start_train_time = time.time()
        train_detector(model=self.train_model,
                       dataset=[mm_train_dataset],
                       cfg=self.config_manager.config_copy,
                       validate=True)
        training_duration = time.time() - start_train_time
        return training_duration

    def _load_best_model_and_check_if_model_improved(self, pretraining_performance: float,
                                                     compare_pre_and_post_training_performance: bool):
        """
        Load the best model from the best_mAP checkpoint, and checks if the model has improved if necessary

        :param pretraining_performance: float, performance of the model on the validation set before training
        :param compare_pre_and_post_training_performance: bool, whether or not to compare performance
        :return (best_score, best_checkpoint, improved): (float, str, bool)
            best_score: the best score of the model after training
            improved: whether or not the score is higher than the before-training model
        """
        # Load the best model from the best_mAP checkpoint
        last_checkpoint = torch.load(os.path.join(self.config_manager.config.work_dir, 'latest.pth'))
        best_checkpoint = torch.load(os.path.join(self.config_manager.config.work_dir,
                                                  last_checkpoint['meta']['hook_msgs']['best_ckpt']))
        best_score = last_checkpoint['meta']['hook_msgs']['best_score']

        # Check whether model has improved
        improved = False
        if compare_pre_and_post_training_performance:
            if best_score > pretraining_performance:
                improved = True

        # Load the best weights
        self.train_model.load_state_dict(best_checkpoint['state_dict'])
        return best_score, improved

    def _persist_new_model(self, dataset: Dataset, performance: Performance, training_duration: float):
        """
        Convert mmdetection model to NOUS model and persist into database. Also update inference model for task

        :param dataset: NOUS dataset that was used for training
        :param performance: performance metrics of the model
        :param training_duration: duration of the training round
        """
        # First make sure train_model.cfg is up to date, then load state_dict and config to bytes in model_data
        self.train_model.cfg = self.config_manager.config_copy
        model_data = self.get_model_bytes()
        model = Model(project=self.task_environment.project,
                      task_node=self.task_environment.task_node,
                      configuration=self.task_environment.get_model_configuration(),
                      data=model_data,
                      tags=None,
                      performance=performance,
                      train_dataset=dataset,
                      training_duration=training_duration)

        self.task_environment.model = model
        self.inference_model = copy.deepcopy(self.train_model)
        self.inference_model.eval()

    def train(self, dataset: Dataset, train_parameters: Optional[TrainParameters] = None) -> Model:
        """ Trains a model on a dataset """
        # Configure datasets
        self.config_manager.update_dataset_subsets(dataset=dataset, model=None)
        # Dataset building modifies the config in place, so use a copy
        mm_train_dataset = build_dataset(self.config_manager.config_copy.data.train)

        # Create a directory to store model checkpoints for this training round. Also writes the model config to a file
        # 'config.py' in that directory, for debugging purposes.
        train_round_checkpoint_dir = self._create_training_checkpoint_dirs()

        # Create new train_model if training from scratch
        old_train_model, train_from_scratch = self._is_train_from_scratch(train_parameters)

        # Evaluate model performance before training
        pretraining_performance, compare_performance = self._do_pre_evaluation(dataset)

        # Check for stop signal between pre-eval and training. If training is cancelled at this point, old_train_model
        # should be restored when training from scratch.
        if self.should_stop:
            self.should_stop = False
            if train_from_scratch:
                self.train_model = old_train_model
            return self.task_environment.model

        # Train the model
        training_duration = self._do_model_training(mm_train_dataset)

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        # model should be returned. Old train model is restored.
        if self.should_stop:
            self.should_stop = False
            logger.info('Training cancelled.')
            self.train_model = old_train_model
            return self.task_environment.model

        # Load the best weights and check if model has improved
        best_score, improved = self._load_best_model_and_check_if_model_improved(pretraining_performance,
                                                                                 compare_performance)
        # Return a new model if model has improved, or there is no model yet.
        if improved or isinstance(self.task_environment.model, NullModel):
            if improved:
                logger.info("Training finished, and it has an improved model")
            else:
                logger.info("First training round, NOUS is saving the model.")
            # Add mAP metric and loss curves
            performance = Performance(score=ScoreMetric(value=best_score, name="mAP"),
                                      dashboard_metrics=self.generate_training_metrics_group())
            self._persist_new_model(dataset, performance, training_duration)
        else:
            logger.info("Model performance has not improved while training. No new model has been saved.")
            if train_from_scratch:
                # Restore old training model if training from scratch and not improved
                self.train_model = old_train_model

        self.is_training = False
        return self.task_environment.model

    def _load_current_training_logs(self):
        """
        Loads the logs of the last training run.

        :return log_lines: list of log entries. Each entry is a dict of log variables.
        """
        training_logs = glob.glob(os.path.join(self.config_manager.config.work_dir, '*.log.json'))
        if not training_logs:
            return None

        with open(training_logs[0], 'r') as f:
            log_lines = []
            for line in f:
                log_lines.append(json.loads(line))

        if log_lines:
            # First line just contains experiment name, so drop that one
            log_lines.pop(0)

        return log_lines

    def get_training_progress(self) -> float:
        """
        Calculate the progress of the current training
        # TODO: Port TimeMonitorCallback to mmcv Hook, to implement this in a nice way that doesn't require parsing logs
        #   JIRA ticket:
        #   https://cosmonio.atlassian.net/browse/NI-661?atlOrigin=eyJpIjoiOWU2YjY5NmQwOGFiNDdlMTg3YzA1YzYxMGZjYTNiNzQiLCJwIjoiaiJ9

        :return: training progress in percent
        """
        log_entries = self._load_current_training_logs()
        if (not log_entries) or (not self.is_training):
            return 0
        total_epochs = self.config_manager.config.runner.max_epochs
        batch_size = self.config_manager.config.data.samples_per_gpu
        iterations_per_epoch = int(np.ceil(self.n_samples_in_current_training_set/batch_size))
        total_iterations = iterations_per_epoch*total_epochs

        current_epoch = log_entries[-1].get('epoch', 1) - 1
        current_iter = log_entries[-1].get('iter', 1)

        return float((current_iter+(current_epoch*iterations_per_epoch))/total_iterations * 100)

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
        result_based_nms_threshold = params.postprocessing.result_based_nms_threshold.value
        cross_class_nms = params.postprocessing.cross_class_nms.value

        f_measure_metrics = MetricsHelper.compute_f_measure(resultset,
                                                            result_based_confidence_threshold,
                                                            result_based_nms_threshold,
                                                            cross_class_nms)

        if resultset.purpose is ResultsetPurpose.EVALUATION:
            # only set configurable params based on validation result set
            if result_based_confidence_threshold:
                best_confidence_threshold = f_measure_metrics.best_confidence_threshold
                if best_confidence_threshold is not None:
                    logger.info(f"Setting confidence_threshold to " f"{best_confidence_threshold.value} based on results")
                    params.postprocessing.confidence_threshold.value = best_confidence_threshold.value
                else:
                    raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")

            if result_based_nms_threshold:
                best_nms_threshold = f_measure_metrics.best_nms_threshold
                if best_nms_threshold is not None:
                    logger.info(f"Setting nms_threshold to {best_nms_threshold.value} based on results")
                    params.postprocessing.nms_threshold.value = best_nms_threshold.value
                else:
                    raise ValueError(f"Cannot compute metrics: Invalid confidence threshold!")
            self.task_environment.set_configurable_parameters(params)

        logger.info(f"F-measure after evaluation: {f_measure_metrics.f_measure.value}")

        return f_measure_metrics.get_performance()

    def generate_training_metrics_group(self) -> Optional[List[MetricsGroup]]:
        """
        Parses the mmdetection logs to get metrics from the latest training run

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []
        log_list = self._load_current_training_logs()
        if not log_list:
            return None
        # Last line contains duplicate validation epoch due to save_best checkpoint mechanism, drop that one
        log_list.pop(-1)

        train_entries = [x for x in log_list if x.get('mode', None) == 'train']
        val_entries = [x for x in log_list if x.get('mode', None) == 'val']
        n_val_epochs = max([x.get('epoch', 0) for x in val_entries])

        if train_entries:
            n_train_epochs = max([x.get('epoch', 0) for x in train_entries])
            max_train_iters = max([x.get('iter', 0) for x in train_entries])
        else:
            n_train_epochs = n_val_epochs
            max_train_iters = 0

        real_epochs = min([n_train_epochs, n_val_epochs])

        df = pd.DataFrame(index=np.arange(1, real_epochs+1, 1), columns=['val_mAP', 'train_loss', 'learning_rate'])
        for entry in log_list:
            if entry.get('mode', None) == 'val':
                df.at[entry['epoch'], 'val_mAP'] = entry.get('mAP', 0)
                df.at[entry['epoch'], 'learning_rate'] = entry.get('lr', 0)
            elif entry.get('mode', None) == 'train':
                if entry['iter'] == max_train_iters:
                    df.at[entry['epoch'], 'train_loss'] = entry.get('loss', 0)

        # Model architecture
        architecture = InfoMetric(name='Model architecture', value=self.train_model.cfg.model.type)
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

        # Training loss and learning rate, if the dataset is really small so that there are less than 5 iterations in
        # one epoch this will not be logged, so in that case we do not add it to the metrics.
        if train_entries:
            # Training loss
            loss = CurveMetric(ys=df['train_loss'], name="Training")
            visualization_info = LineChartInfo(name="Loss curve", x_axis_label="Epoch", y_axis_label="Loss value")
            output.append(MetricsGroup(metrics=[loss], visualization_info=visualization_info))

            # Learning rate
            lr = CurveMetric(ys=df['learning_rate'], name="Learning rate")
            visualization_info_lr = LineChartInfo(name="Learning rate", x_axis_label="Epoch",
                                                  y_axis_label="Learning rate")
            output.append(MetricsGroup(metrics=[lr], visualization_info=visualization_info_lr))

        # mAP score
        val_map = CurveMetric(ys=df['val_mAP'], name="Validation")
        visualization_info_map = LineChartInfo(name="Mean Average Precision (mAP)", x_axis_label="Epoch",
                                               y_axis_label="Mean Average Precision")
        output.append(MetricsGroup(metrics=[val_map], visualization_info=visualization_info_map))

        return output

    def get_model_bytes(self) -> bytes:
        """
        Returns the data of the current model. We store both the state_dict and the configuration to make the model
        self-contained.

        :return: {'state_dict': data of current model in bytes, 'config': mmdetection config string used for training}
        """
        buffer = io.BytesIO()
        config_str = self.config_manager.config_to_string(self.train_model.cfg)
        torch.save({'state_dict': self.train_model.state_dict(), 'config': config_str},
                   buffer)
        return bytes(buffer.getbuffer())

    @staticmethod
    def get_configurable_parameters(task_environment: TaskEnvironment) -> MMDetectionParameters:
        """
        Returns the configurable parameters.

        :param task_environment: Current task environment
        :return: Instance of MMDetectionParameters
        """
        return task_environment.get_configurable_parameters(instance_of=MMDetectionParameters)

    def update_configurable_parameters(self, task_environment: TaskEnvironment):
        """
        Called when the user changes the configurable parameters in the UI.

        :param task_environment: New task environment with updated configurable parameters
        """
        previous_conf_params = self.get_configurable_parameters(self.task_environment)
        new_conf_params = self.get_configurable_parameters(task_environment)

        self.task_environment = task_environment
        self.config_manager.update_project_configuration(new_conf_params)

        # Check whether model architecture has changed, because then models have to be loaded again
        previous_model_arch = previous_conf_params.learning_architecture.model_architecture.value
        new_model_arch = new_conf_params.learning_architecture.model_architecture.value
        if new_model_arch != previous_model_arch:
            self.load_model(task_environment)

    def get_confidence_and_nms_thresholds(self, is_evaluation: bool) -> Tuple[float, float, bool]:
        """
        Retrieves the thresholds for confidence and non maximum suppression (nms) from the configurable parameters. If
        is_evaluation is True, the confidence threshold is set to 0 and nms threshold is set to 1, since in that case
        all detections are accepted in order to compute optimum values for the thresholds. Also returns whether or not
        to perform nms across objects of different classes.

        :param is_evaluation: bool, True in case analysis is requested for evaluation

        :return confidence_threshold: float, threshold for prediction confidence
        :return nms_threshold: float, threshold for non maximum suppression
        :return cross_class_nms: bool, True in case cross_class_nms should be performed
        """
        conf_params = self.get_configurable_parameters(self.task_environment)
        confidence_threshold = conf_params.postprocessing.confidence_threshold.value
        nms_threshold = conf_params.postprocessing.nms_threshold.value
        result_based_confidence_threshold = conf_params.postprocessing.result_based_confidence_threshold.value
        result_based_nms_threshold = conf_params.postprocessing.result_based_nms_threshold.value
        cross_class_nms = conf_params.postprocessing.cross_class_nms.value
        if is_evaluation:
            if result_based_confidence_threshold:
                confidence_threshold = 0.0
            if result_based_nms_threshold:
                nms_threshold = 1.0
        return confidence_threshold, nms_threshold, cross_class_nms

    @staticmethod
    def is_docker():
        """
        Checks whether the task runs in docker container

        :return bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        return (
                os.path.exists('/.dockerenv') or
                os.path.isfile(path) and any('docker' in line for line in open(path))
        )

    def unload(self):
        """
        Unload the task
        """
        self.delete_scratch_space()
        if self.is_docker():
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
        optimized_models = [self.generate_openvino_model()]
        return optimized_models

    def prepare_trained_model_for_onnx_conversion(self, optimized_model_dir: str):
        """
        Prepares the model for conversion to ONNX format. The conversion mechanism from mmdetection is file system based
        so it uses the model configuration and model checkpoint stored in the most recent temporary training directory.

        :param optimized_model_dir: directory in which the sample image for ONNX tracing will be stored
        :return tuple(model, tensor_data): model that can be passed to onnx exporter, along with input data in
            tensor_data
        """
        # Prepare candidate tracing image. Has to be a real image so that nms post processing operation can be traced
        image_path = None
        for dataset_item in self.task_environment.model.train_dataset:
            if dataset_item.get_shapes(include_empty=False):
                # If the image has objects, run inference to check that the model makes predictions so that nms will
                # be performed. Note that it shouldn't take long to find a valid image, since there is no
                # threshold on probability so any image that will yield even unlikely detections is good.
                inference_results = safe_inference_detector(self.inference_model, dataset_item.numpy)
                if inference_results:
                    # If there are detections, save the image
                    image_path = os.path.join(optimized_model_dir, 'dummy.jpg')
                    mmcv.imwrite(dataset_item.numpy, image_path)
                    break

        if image_path is None:
            raise ValueError('Unable to find valid image for ONNX tracing.')

        width, height = self.config_manager.input_image_dimensions
        channels = self.in_channels

        work_dir = self.config_manager.config.work_dir
        input_config = {'input_shape': (1, channels, height, width),
                        'input_path': image_path,
                        'normalize_cfg': self.config_manager.config.img_norm_cfg}
        config_path = os.path.join(work_dir, 'config.py')
        checkpoint_path = os.path.join(work_dir, 'best_mAP.pth')
        if not os.path.isfile(checkpoint_path):
            # If for whatever reason the checkpoint doesn't exist, create it from the current inference model. The
            # checkpoint has to be on disk in order to use the mmdet generate_inputs_and_wrap_model utility.
            save_checkpoint(self.inference_model, checkpoint_path)
        if not os.path.isfile(config_path):
            # If config path doesn't exists, create it from current inference model. This can occur if task unloading is
            # called right after training, since that deletes the scratch space where the config lives.
            config_path = os.path.join(optimized_model_dir, 'config.py')
            config_string = self.config_manager.config_to_string(copy.deepcopy(self.inference_model.cfg))
            with open(config_path, 'w') as f:
                f.write(config_string)
        model, tensor_data = generate_inputs_and_wrap_model(config_path, checkpoint_path, input_config)
        return model, tensor_data

    def generate_openvino_model(self) -> OpenVINOModel:
        """
        Convert the current model to OpenVINO with FP16 precision by first converting to ONNX and
        then converting the ONNX model to an OpenVINO IR model.
        """
        optimized_model_precision = Precision.FP16

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "mmdetection")
            os.makedirs(optimized_model_dir, exist_ok=True)

            # Convert PyTorch model to ONNX
            onnxmodel_path = os.path.join(optimized_model_dir, 'inference_model.onnx')
            model, tensor_data = self.prepare_trained_model_for_onnx_conversion(optimized_model_dir)
            torch.onnx.export(model, args=tensor_data, f=onnxmodel_path, opset_version=11)
            logger.info("Model conversion to ONNX format was successful.")

            width, height = self.config_manager.input_image_dimensions
            channels = self.in_channels

            # Convert ONNX model to OpenVINO
            parameters = f'--input_model "{optimized_model_dir}/inference_model.onnx" \
                           --data_type {optimized_model_precision.name} \
                           --input_shape "(1,{channels},{height},{width})"'

            logger.info(parameters)
            return generate_openvino_model(parameters=parameters, project=self.task_environment.project,
                                           model=self.task_environment.model,
                                           precision=optimized_model_precision)

    def delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space, ignore_errors=False)
