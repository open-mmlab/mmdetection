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


from .config_utils import (config_from_string, config_to_string, patch_config,
                           prepare_for_testing, prepare_for_training,
                           save_config_to_file, set_hyperparams,
                           set_values_as_default)
from .configuration import OTEDetectionConfig
from .inference_task import OTEDetectionInferenceTask
from .nncf_task import OTEDetectionNNCFTask
from .openvino_task import OpenVINODetectionTask
from .ote_utils import generate_label_schema, get_task_class, load_template
from .train_task import OTEDetectionTrainingTask

__all__ = [
    config_from_string,
    config_to_string,
    generate_label_schema,
    get_task_class,
    load_template,
    OpenVINODetectionTask,
    OTEDetectionConfig,
    OTEDetectionInferenceTask,
    OTEDetectionNNCFTask,
    OTEDetectionTrainingTask,
    patch_config,
    prepare_for_testing,
    prepare_for_training,
    save_config_to_file,
    set_hyperparams,
    set_values_as_default,
    ]
