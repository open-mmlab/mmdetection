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
import os
import tempfile
from typing import Optional, List

from mmcv import Config, ConfigDict
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.label import Label
from sc_sdk.logging import logger_factory
from sc_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback

from .configuration import OTEDetectionConfig


logger = logger_factory.get_logger("OTEDetectionTask")


def apply_template_configurable_parameters(params: OTEDetectionConfig, template: dict):

    def xset(obj, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                xset(getattr(obj, k), v)
            else:
                # if hasattr(getattr(obj, k), 'value'):
                #     getattr(obj, k).value = type(getattr(obj, k).value)(v)
                # else:
                setattr(obj, k, v)

    hyper_params = template['hyper_parameters']['params']
    xset(params, hyper_params)
    params.algo_backend.model_name = template['name']


def patch_config(config: Config, work_dir: str, labels: List[Label], random_seed: Optional[int] = None):
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
        remove_from_config(config, 'total_epochs')

    # Remove high level data pipelines definition leaving them only inside `data` section.
    remove_from_config(config, 'train_pipeline')
    remove_from_config(config, 'test_pipeline')

    # Patch data pipeline, making it OTE-compatible.
    patch_datasets(config)

    config.log_config.hooks = []
    evaluation_metric = config.evaluation.get('metric')
    if evaluation_metric is not None:
        config.evaluation.save_best = evaluation_metric
    config.evaluation.rule = 'greater'

    label_names = [lab.name for lab in labels]
    set_data_classes(config, label_names)

    config.gpu_ids = range(1)
    config.work_dir = work_dir
    config.seed = random_seed


def set_hyperparams(config: Config, hyperparams: OTEDetectionConfig):
    config.optimizer.lr = float(hyperparams.learning_parameters.learning_rate)
    config.lr_config.warmup_iters = int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    config.data.samples_per_gpu = int(hyperparams.learning_parameters.batch_size)
    config.data.workers_per_gpu = int(hyperparams.learning_parameters.num_workers)
    total_iterations = int(hyperparams.learning_parameters.num_iters)
    if 'IterBased' in config.runner.type:
        config.runner.max_iters = total_iterations
    else:  # Epoch based runner
        config.runner.max_epochs = total_iterations
    num_checkpoints = int(hyperparams.learning_parameters.num_checkpoints)
    config.evaluation.interval = total_iterations // num_checkpoints
    config.checkpoint_config.interval = total_iterations // num_checkpoints


def prepare_for_testing(config: Config, dataset: Dataset) -> Config:
    config = copy.deepcopy(config)
    # FIXME. Should working directories be modified here?
    config.data.test.ote_dataset = dataset
    return config


def prepare_for_training(config: Config, train_dataset: Dataset, val_dataset: Dataset,
                         time_monitor: TimeMonitorCallback, learning_curves: defaultdict) -> Config:
    config = copy.deepcopy(config)

    prepare_work_dir(config)

    # config.data.test.ote_dataset = dataset.get_subset(Subset.TESTING)
    config.data.val.ote_dataset = val_dataset
    if 'ote_dataset' in config.data.train:
        config.data.train.ote_dataset = train_dataset
    else:
        config.data.train.dataset.ote_dataset = train_dataset

    if 'custom_hooks' not in config:
        config.custom_hooks = []
    config.custom_hooks.append({'type': 'OTEProgressHook', 'time_monitor': time_monitor, 'verbose': True})
    config.log_config.hooks.append({'type': 'OTELoggerHook', 'curves': learning_curves})

    return config


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


def save_config_to_file(config: Config):
    """ Dump the full config to a file. Filename is 'config.py', it is saved in the current work_dir. """
    filepath = os.path.join(config.work_dir, 'config.py')
    config_string = config_to_string(config)
    with open(filepath, 'w') as f:
        f.write(config_string)


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
    save_config_to_file(config)
    return train_round_checkpoint_dir


def set_data_classes(config: Config, label_names: List[str]):
    # Save labels in data configs.
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset':
            cfg.dataset.classes = label_names
        else:
            cfg.classes = label_names
        config.data[subset].classes = label_names

    # Set proper number of classes in model's detection heads.
    num_classes = len(label_names)
    if 'roi_head' in config.model:
        if isinstance(config.model.roi_head.bbox_head, List):
            for head in config.model.roi_head.bbox_head:
                head.num_classes = num_classes
        else:
            config.model.roi_head.bbox_head.num_classes = num_classes
    elif 'bbox_head' in config.model:
        config.model.bbox_head.num_classes = num_classes
    # FIXME. ?
    # self.config.model.CLASSES = label_names


def patch_datasets(config: Config):
    assert 'data' in config
    for subset in ('train', 'val', 'test'):
        cfg = config.data[subset]
        if cfg.type == 'RepeatDataset':
            cfg = cfg.dataset
        cfg.type = 'OTEDataset'
        cfg.ote_dataset = None
        remove_from_config(cfg, 'ann_file')
        remove_from_config(cfg, 'img_prefix')
        for pipeline_step in cfg.pipeline:
            if pipeline_step.type == 'LoadImageFromFile':
                pipeline_step.type = 'LoadImageFromOTEDataset'
            elif pipeline_step.type == 'LoadAnnotations':
                pipeline_step.type = 'LoadAnnotationFromOTEDataset'


def remove_from_config(config, key: str):
    if key in config:
        if isinstance(config, Config):
            del config._cfg_dict[key]
        elif isinstance(config, ConfigDict):
            del config[key]
        else:
            raise ValueError(f'Unknown config type {type(config)}')


