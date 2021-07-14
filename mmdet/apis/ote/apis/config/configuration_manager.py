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

from collections import defaultdict
import copy
import glob
from inspect import CO_ASYNC_GENERATOR
import os
import tempfile
import torch
from typing import Optional, List

from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.label import Label
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback

from mmdet.apis.ote.extension.utils.hooks import OTELoggerHook
from mmcv import Config, ConfigDict
from mmcv.utils import get_git_hash
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet.integration.nncf import get_nncf_metadata
from mmdet.integration.nncf.config import load_nncf_config, compose_nncf_config

from ..detection import MMDetectionParameters
from .config_mapper import ConfigMappings
from .task_types import MMDetectionTaskType


logger = logger_factory.get_logger("MMDetectionTask")


class ConfigManager:
    def __init__(self,
                 config: Config,
                 hyperparams: MMDetectionParameters,
                 labels: List[Label],
                 scratch_space: str,
                 random_seed: Optional[int] = 42):

        self.set_config(config)
        self.set_hyperparams(hyperparams)
        self.labels = labels
        self.label_names = [lab.name for lab in labels]
        self.set_data_classes(self.label_names)
        self.config.gpu_ids = range(1)
        self.config.work_dir = scratch_space
        self.config.seed = random_seed

        self.time_monitor = TimeMonitorCallback(0, 0, 0, 0)
        self.learning_curves = defaultdict(OTELoggerHook.Curve)

    def set_config(self, config: Config):
        self.config = config
        self.patch_config(self.config)

    @staticmethod
    def patch_config(config: Config):
        # Set runner if not defined.
        if 'runner' not in config:
            config.runner = {'type': 'EpochBasedRunner'}

        # Check that there is no conflict in specification of number of training epochs.
        # Move global definition of epochs inside runner config.
        if 'total_epochs' in config:
            if config.runner.type == 'EpochBasedRunner':
                if config.runner.max_epochs != config.total_epochs:
                    logger.warning('Conflicting declaration of training epochs number.')
                config.runner.max_epochs = config.total_epochs
            else:
                logger.warning('Total number of epochs set for an iteration based runner.')
            ConfigManager.remove_from_config(config, 'total_epochs')

        # Remove high level data pipelines definition leaving them only inside `data` section.
        ConfigManager.remove_from_config(config, 'train_pipeline')
        ConfigManager.remove_from_config(config, 'test_pipeline')

        # Patch data pipeline, making it OTE-compatible.
        ConfigManager.patch_datasets(config)

        config.log_config.hooks = []
        evaluation_metric = config.evaluation.get('metric')
        if evaluation_metric is not None:
            config.evaluation.save_best = evaluation_metric
        config.evaluation.rule = 'greater'

    @staticmethod
    def patch_datasets(config: Config):
        assert 'data' in config
        for subset in ('train', 'val', 'test'):
            cfg = config.data[subset]
            if cfg.type == 'RepeatDataset':
                cfg = cfg.dataset
            cfg.type = 'OTEDataset'
            cfg.ote_dataset = None
            ConfigManager.remove_from_config(cfg, 'ann_file')
            ConfigManager.remove_from_config(cfg, 'img_prefix')
            for pipeline_step in cfg.pipeline:
                if pipeline_step.type == 'LoadImageFromFile':
                    pipeline_step.type = 'LoadImageFromOTEDataset'
                elif pipeline_step.type == 'LoadAnnotations':
                    pipeline_step.type = 'LoadAnnotationFromOTEDataset'

    def set_data_classes(self, label_names):
        # Save labels in data configs.
        for subset in ('train', 'val', 'test'):
            cfg = self.config.data[subset]
            if cfg.type == 'RepeatDataset':
                cfg.dataset.classes = label_names
            else:
                cfg.classes = label_names
            self.config.data[subset].classes = label_names

        # Set proper number of classes in model's detection heads.
        num_classes = len(label_names)
        if 'roi_head' in self.config.model:
            if isinstance(self.config.model.roi_head.bbox_head, List):
                for head in self.config.model.roi_head.bbox_head:
                    head.num_classes = num_classes
            else:
                self.config.model.roi_head.bbox_head.num_classes = num_classes
        elif 'bbox_head' in self.config.model:
            self.config.model.bbox_head.num_classes = num_classes
        # FIXME. ?
        # self.config.model.CLASSES = label_names

    @staticmethod
    def remove_from_config(config, key: str):
        if key in config:
            if isinstance(config, Config):
                del config._cfg_dict[key]
            elif isinstance(config, ConfigDict):
                del config[key]
            else:
                raise ValueError(f'Unknown config type {type(config)}')

    def set_hyperparams(self, hyperparams: MMDetectionParameters):
        config = self.config
        config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate.value)
        config.lr_config.warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters.value)
        config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size.value)
        total_iterations = int(hyperparams.learning_parameters.num_iters.value)
        if 'IterBased' in config.runner.type:
            config.runner.max_iters = total_iterations
        else:  # Epoch based runner
            config.runner.max_epochs = total_iterations
        config.evaluation.interval = total_iterations // 10
        config.checkpoint_config.interval = total_iterations // 10

    @staticmethod
    def prepare_for_testing(config: Config, dataset: Dataset):
        config = copy.deepcopy(config)
        # FIXME. Should working directories be modified here?
        config.data.test.ote_dataset = dataset.get_subset(Subset.TESTING)
        return config

    # def prepare_for_training(self, dataset: Dataset):
    #     config = self.config_copy

    #     self.prepare_work_dir(config)

    #     # config.data.test.ote_dataset = dataset.get_subset(Subset.TESTING)
    #     config.data.val.ote_dataset = dataset.get_subset(Subset.VALIDATION)
    #     if 'ote_dataset' in config.data.train:
    #         config.data.train.ote_dataset = dataset.get_subset(Subset.TRAINING)
    #     else:
    #         config.data.train.dataset.ote_dataset = dataset.get_subset(Subset.TRAINING)

    #     self.time_monitor = TimeMonitorCallback(0, 0, 0, 0)
    #     self.learning_curves = defaultdict(OTELoggerHook.Curve)
    #     if 'custom_hooks' not in config:
    #         config.custom_hooks = []
    #     config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': self.time_monitor, 'verbose': True})
    #     config.log_config.hooks.append({'type': 'OTELoggerHook', 'curves': self.learning_curves})

    #     return config

    @staticmethod
    def prepare_for_training(config: Config, dataset: Dataset, time_monitor: TimeMonitorCallback, learning_curves: defaultdict):
        config = copy.deepcopy(config)

        ConfigManager.prepare_work_dir(config)

        # config.data.test.ote_dataset = dataset.get_subset(Subset.TESTING)
        config.data.val.ote_dataset = dataset.get_subset(Subset.VALIDATION)
        if 'ote_dataset' in config.data.train:
            config.data.train.ote_dataset = dataset.get_subset(Subset.TRAINING)
        else:
            config.data.train.dataset.ote_dataset = dataset.get_subset(Subset.TRAINING)

        if 'custom_hooks' not in config:
            config.custom_hooks = []
        config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': time_monitor, 'verbose': True})
        config.log_config.hooks.append({'type': 'OTELoggerHook', 'curves': learning_curves})

        return config

    @property
    def config_copy(self):
        return copy.deepcopy(self.config)

    @staticmethod
    def config_to_string(config: Config) -> str:
        """
        Convert a full mmdetection config to a string.

        :param config: configuration object to convert
        :return str: string representation of the configuration
        """
        config_copy = copy.deepcopy(config)
        # Clean config up by removing dataset and label entities as this causes the pretty text parsing to fail
        config_copy.data.test.ote_dataset = None
        config_copy.data.val.ote_dataset = None
        if 'ote_dataset' in config_copy.data.train:
            config_copy.data.train.ote_dataset = None
        else:
            config_copy.data.train.dataset.ote_dataset = None
        # config_copy.labels = [label.name for label in config.labels]
        return Config(config_copy).pretty_text

    @staticmethod
    def config_from_string(config_string: str) -> Config:
        """
        Generate an mmdetection config dict object from a string.

        :param config_string: string to parse
        :return config: configuration object
        """
        with tempfile.NamedTemporaryFile('w', suffix='.py') as temp_file:
            temp_file.write(config_string)
            temp_file.flush()
            return Config.fromfile(temp_file.name)

    @staticmethod
    def save_config_to_file(config: Config):
        """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """
        filepath = os.path.join(config.work_dir, 'config.py')
        config_string = ConfigManager.config_to_string(config)
        with open(filepath, 'w') as f:
            f.write(config_string)

    @staticmethod
    def prepare_work_dir(config: Config) -> str:
        base_work_dir = config.work_dir
        checkpoint_dirs = glob.glob(os.path.join(base_work_dir, "checkpoints_round_*"))
        train_round_checkpoint_dir = os.path.join(base_work_dir, f"checkpoints_round_{len(checkpoint_dirs)}")
        os.makedirs(train_round_checkpoint_dir)
        logger.info(f"Checkpoints and logs for this training run are stored in {train_round_checkpoint_dir}")
        config.work_dir = train_round_checkpoint_dir
        if 'meta' not in config.runner:
            config.runner.meta = ConfigDict()
        config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"
        # Save training config for debugging. It is saved in the checkpoint dir for this training round
        ConfigManager.save_config_to_file(config)
        return train_round_checkpoint_dir


class MMDetectionConfigManager(object):
    def __init__(self, task_environment: TaskEnvironment, task_type: MMDetectionTaskType, scratch_space: str,
                 random_seed: Optional[int] = 42):
        """
        Class that configures an mmdetection model and training configuration. Initializes the task-specific
        configuration. Sets the work_dir for mmdetection and the number of classes in the model. Also seeds random
        generators.

        :param task_environment: Task environment for the task, containing configurable parameters, labels, etc.
        :param task_type: MMDetectionTaskType of the task at hand
        :param scratch_space: Path to working directory
        :param random_seed: Optional int to seed random generators.
        """
        # Initialize configuration mappings for the task type and get configurable parameters
        self.config_mapper = ConfigMappings()
        conf_params = task_environment.get_configurable_parameters(instance_of=MMDetectionParameters)

        # Build the config
        template = conf_params.algo_backend.template.value
        self.model_name = conf_params.algo_backend.model_name.value
        base_dir = os.path.abspath(os.path.dirname(template))
        model_config = os.path.join(base_dir, conf_params.algo_backend.model.value)
        data_pipeline = os.path.join(base_dir, conf_params.algo_backend.data_pipeline.value)

        self.custom_lr_schedule = self._get_custom_lr_schedule(model_config)
        self.nncf_config = load_nncf_config(os.path.join(base_dir, conf_params.learning_parameters.nncf_config.value))

        logger.warning(f'model config {model_config}')
        logger.warning(f'data pipeline {data_pipeline}')

        self._compose_config(
            model_file=model_config,
            schedule_file=None,
            dataset_file=data_pipeline,
            runtime_file=self.config_mapper.get_runtime_file('default')
        )

        # Fix config.
        if hasattr(self.config, 'total_epochs'):
            self.config.runner.max_epochs = self.config.total_epochs

        # Assign additional parameters
        # FIXME.
        self.config.gpu_ids = range(1)

        # this controls the maximum number of ground truth bboxes per image that will be processed on the gpu. If an
        # image contains more gt bboxes than this, they will be moved to the cpu for processing. It is set to avoid
        # gpu oom errors
        self._max_number_gt_bbox_per_image_on_gpu = 100

        # mmdetection training needs workdir to store logs and checkpoints
        self.config.work_dir = scratch_space
        self.config.seed = random_seed
        set_random_seed(random_seed)

        # Specify label names in config
        labels = task_environment.labels
        self.label_names = [lab.name for lab in labels]
        # FIXME. What for?
        self.config.labels = labels
        self.set_data_classes()

        # Finally, update the config to make sure the model heads have the correct number of classes, and the values
        # set in the configurable parameters are reflected in the config
        self._update_model_classification_heads()
        self.update_project_configuration(conf_params)

    def _update_nncf_config_section(self, configurable_parameters: MMDetectionParameters):
        enabled_nncf_options = []
        AVAILABLE_NNCF_OPTIONS = ('nncf_quantization', 'nncf_sparsity', 'nncf_pruning', 'nncf_binarization')
        for option in AVAILABLE_NNCF_OPTIONS:
            flag = configurable_parameters.learning_parameters[option].value
            if flag:
                enabled_nncf_options.append(option)

        if 'nncf_config' in self.config:
            del self.config.nncf_config
        if len(enabled_nncf_options) > 0:
            nncf_config = compose_nncf_config(self.nncf_config, enabled_nncf_options)
            # FIXME. NNCF configuration may override some training parameters, like number of epochs.
            config = Config._merge_a_into_b(nncf_config, self.config)
            self.config = Config(config)

            if self.config.checkpoint_config is not None:
                # save mmdet version, config file content and class names in
                # checkpoints as meta data
                self.config.checkpoint_config.meta = dict(
                    mmdet_version=mmdet_version + get_git_hash()[:7],
                    CLASSES=self.label_names)
                # also save nncf status in the checkpoint -- it is important,
                # since it is used in wrap_nncf_model for loading NNCF-compressed models
                nncf_metadata = get_nncf_metadata()
                self.config.checkpoint_config.meta.update(nncf_metadata)
            else:
                # cfg.checkpoint_config is None
                assert not self.config.get('nncf_config'), (
                        "NNCF is enabled, but checkpoint_config is not set -- "
                        "cannot store NNCF metainfo into checkpoints")

    def _get_custom_lr_schedule(self, model_file: str):
        schedule_sections = ('optimizer', 'optimizer_config', 'lr_config', 'momentum_config')
        model_config = Config.fromfile(model_file)
        schedule_config = dict()
        for section in schedule_sections:
            if section in model_config:
                schedule_config[section] = model_config[section]
        return schedule_config

    def _compose_config(self, model_file: str, schedule_file: str, dataset_file: str, runtime_file: str):
        """
        Constructs the full mmdetection configuration from files containing the different config sections

        :param model_file: Path to the model config file
        :param schedule_file: Path to the learning rate schedule file
        :param dataset_file: Path to the dataset config file
        :param runtime_file: Path to the runtime config file
        """
        config_file_list = [model_file, schedule_file, dataset_file, runtime_file]
        config = dict()
        for filename in config_file_list:
            if filename is None:
                continue
            update_config = Config.fromfile(filename)
            config = Config._merge_a_into_b(update_config, config)
        self.config = Config(config)

    def set_data_classes(self):
        """ Sets the label names for the different subsets """
        subsets = ['train', 'val', 'test']
        for subset in subsets:
            cfg = self.config.data[subset]
            if cfg.type == 'RepeatDataset':
                cfg.dataset.classes = self.label_names
            else:
                cfg.classes = self.label_names
            # self.config.data[subset].classes = self.label_names

    def update_project_configuration(self, configurable_parameters: MMDetectionParameters):
        """
        Update the mmdetection model configuration according to the configurable parameters.

        :param configurable_parameters: Parameters to set

        """
        learning_rate_schedule_name = configurable_parameters.learning_parameters.learning_rate_schedule.value
        learning_rate_warmup_iters = configurable_parameters.learning_parameters.learning_rate_warmup_iters.value
        self._update_learning_rate_schedule(learning_rate_schedule_name, learning_rate_warmup_iters)
        if 'IterBased' in self.config.runner.type:
            self.config.runner.max_iters = int(configurable_parameters.learning_parameters.num_iters.value)
        else:
            self.config.runner.max_epochs = int(configurable_parameters.learning_parameters.num_iters.value)
        self.config.optimizer.lr = float(configurable_parameters.learning_parameters.learning_rate.value)
        self.config.data.samples_per_gpu = int(configurable_parameters.learning_parameters.batch_size.value)
        self._update_nncf_config_section(configurable_parameters)

    def update_dataset_subsets(self, dataset: Dataset, model: torch.nn.Module = None):
        """
        Set the correct dataset subsets in an mmdetection configuration

        :param dataset: Dataset that defines the subsets
        :param model: If a model is passed, the config of that model will be updated instead of the config maintained
            by the config_manager.

        :return: model with updated data configuration
        """
        if model is None:
            cfg_to_change = self.config.data
        else:
            cfg_to_change = model.cfg.data

        cfg_to_change.test.ote_dataset = dataset.get_subset(Subset.TESTING)
        cfg_to_change.val.ote_dataset = dataset.get_subset(Subset.VALIDATION)
        if 'ote_dataset' in cfg_to_change.train:
            cfg_to_change.train.ote_dataset = dataset.get_subset(Subset.TRAINING)
        else:
            cfg_to_change.train.dataset.ote_dataset = dataset.get_subset(Subset.TRAINING)
        return model

    @property
    def config_copy(self):
        """
        Return a copy of the config, for passing to certain mmdetection methods that modify config in place, such
        as train_detector

        :return Config:
        """
        return copy.deepcopy(self.config)

    @staticmethod
    def config_to_string(config: Config) -> str:
        """
        Convert a full mmdetection config to a string.

        :param config: configuration object to convert
        :return str: string representation of the configuration
        """
        config_copy = copy.deepcopy(config)
        # Clean config up by removing dataset and label entities as this causes the pretty text parsing to fail
        config_copy.data.test.ote_dataset = None
        config_copy.data.val.ote_dataset = None
        if 'ote_dataset' in config_copy.data.train:
            config_copy.data.train.ote_dataset = None
        else:
            config_copy.data.train.dataset.ote_dataset = None
        config_copy.labels = [label.name for label in config.labels]
        return Config(config_copy).pretty_text

    @staticmethod
    def config_from_string(config_string: str) -> Config:
        """
        Generate an mmdetection config dict object from a string.

        :param config_string: string to parse
        :return config: configuration object
        """
        with tempfile.NamedTemporaryFile('w', suffix='.py') as temp_file:
            temp_file.write(config_string)
            temp_file.flush()
            return Config.fromfile(temp_file.name)

    def save_config_to_file(self):
        """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """
        filepath = os.path.join(self.config.work_dir, 'config.py')
        config_string = self.config_to_string(self.config)
        with open(filepath, 'w') as f:
            f.write(config_string)

    def prepare_work_dir(self, base_work_dir) -> str:
        """
        Create directory to store checkpoints for next training run. Also sets experiment name and updates config

        :return train_round_checkpoint_dir: str, path to checkpoint dir
        """
        checkpoint_dirs = glob.glob(os.path.join(base_work_dir, "checkpoints_round_*"))
        train_round_checkpoint_dir = os.path.join(base_work_dir, f"checkpoints_round_{len(checkpoint_dirs)}")
        os.makedirs(train_round_checkpoint_dir)
        logger.info(f"Checkpoints and logs for this training run are stored in {train_round_checkpoint_dir}")
        self.config.work_dir = train_round_checkpoint_dir
        self.config.runner.meta.exp_name = f"train_round_{len(checkpoint_dirs)}"
        # Save training config for debugging. It is saved in the checkpoint dir for this training round
        self.save_config_to_file()
        return train_round_checkpoint_dir

    def _replace_config_section_from_file(self, file) -> Config:
        """
        Replace part of the configuration by a config file.

        :param file: Config file containing the config section to update
        :return Config: config section that was updated
        """
        config = self.config_copy
        config_section = Config.fromfile(file)
        new_config = Config._merge_a_into_b(config_section, config)
        self.config = Config(new_config)
        return config_section

    def _update_learning_rate_schedule(self, schedule_name: str, warmup_iters: int):
        """
        Update the learning rate scheduling config section in the current configuration

        :param schedule_file: Path to the learning rate schedule file containing the desired schedule
        """

        # remove old optimizer and lr config sections
        sections_to_pop = ('optimizer', 'optimizer_config', 'lr_config', 'momentum_config')
        for section in sections_to_pop:
            if section in self.config:
                self.config.pop(section)

        if schedule_name == 'custom':
            for section in sections_to_pop:
                if section in self.custom_lr_schedule:
                    self.config[section] = self.custom_lr_schedule[section]
        else:
            schedule_file = self.config_mapper.get_schedule_file(schedule_name)
            logger.warning(f'Update LR schedule from {schedule_file}')
            self._replace_config_section_from_file(schedule_file)

        # Set gradient clipping if required for the model in config
        # self._update_gradient_clipping()

        # Set learning rate warmup settings.
        if warmup_iters > 0:
            self.config.lr_config.warmup = 'linear'
            self.config.lr_config.warmup_ratio = 1.0 / 3
            self.config.lr_config.warmup_iters = warmup_iters

    def _update_model_classification_heads(self):
        """ Modify the number of classes of the model in the box heads """
        if 'roi_head' in self.config.model.keys():
            if isinstance(self.config.model.roi_head.bbox_head, List):
                for head in self.config.model.roi_head.bbox_head:
                    head.num_classes = len(self.label_names)
            else:
                self.config.model.roi_head.bbox_head.num_classes = len(self.label_names)
        elif 'bbox_head' in self.config.model.keys():
            self.config.model.bbox_head.num_classes = len(self.label_names)

    def get_lr_schedule_friendly_name(self, lr_policy_type: str):
        """
        Gives the user friendly name of the learning rate schedule associated with lr_policy_type

        :param lr_policy_type: name of the learning rate policy type
        :return: str, friendly name of this learning rate schedule
        """
        return self.config_mapper.get_schedule_friendly_name(lr_policy_type)

    def _search_in_config_dict(self, config_dict, key_to_search, prior_keys=None, results=None):
        """
        Recursively searches a config_dict for all instances of key_to_search and returns the key path to them
        :param config_dict: dict to search
        :param key_to_search: dict key to look for
        :return (value_at_key_to_search, key_path_to_key_to_search)
        """
        if prior_keys is None:
            prior_keys = list()
        if results is None:
            results = []
        if isinstance(config_dict, List):
            dict_to_search_in = {k: v for k, v in enumerate(config_dict)}
        else:
            dict_to_search_in = config_dict
        if not isinstance(dict_to_search_in, dict):
            return results
        for key, value in dict_to_search_in.items():
            current_key_path = prior_keys + [key]
            if key == key_to_search:
                results.append([value, prior_keys])
            self._search_in_config_dict(value, key_to_search, current_key_path, results)
        return results
