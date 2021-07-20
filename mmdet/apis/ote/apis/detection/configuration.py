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

import attr
from attr import attrs

from sc_sdk.configuration import TaskConfig, UIRules, Rule, Operator, Action, ModelLifecycle
from sc_sdk.configuration.config_element_type import ElementCategory
from sc_sdk.configuration.elements import configurable_float, configurable_integer, configurable_boolean, selectable, \
    float_selectable, ParameterGroup, string_attribute, add_parameter_group
from sc_sdk.configuration.elements.primitive_parameters import set_common_metadata
from sc_sdk.configuration.ui_rules import UIRules, NullUIRules


class StringAttr:
    category = ElementCategory.PRIMITIVES


def configurable_str(default_value: str,
                     header: str,
                     description: str = 'Default integer description',
                     warning: str = None,
                     editable: bool = True,
                     visible_in_ui: bool = True,
                     affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
                     ui_rules: UIRules = NullUIRules()) -> int:
    metadata = set_common_metadata(default_value=default_value, header=header, description=description, warning=warning,
                                   editable=editable, visible_in_ui=visible_in_ui, ui_rules=ui_rules,
                                   affects_outcome_of=affects_outcome_of, type=StringAttr)
    return attr.ib(default=default_value,
                   type=int,
                   metadata=metadata)


@attrs
class OTEDetectionConfig(TaskConfig):
    header = string_attribute("Configuration for an object detection task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        num_iters = configurable_integer(
            default_value=10000,
            min_value=10,
            max_value=100000,
            header="Number of training iterations",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate = configurable_float(
            default_value=0.01,
            min_value=1e-07,
            max_value=1e-01,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        learning_rate_warmup_iters = configurable_integer(
            default_value=100,
            min_value=0,
            max_value=10000,
            header="Number of iterations for learning rate warmup",
            description="",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        num_workers = configurable_integer(
            default_value=0,
            min_value=0,
            max_value=8,
            header="Number of cpu threads to use during batch generation",
            description="Increasing this value might improve training speed however it might cause out of memory "
                        "errors. If the number of workers is set to zero, data loading will happen in the main "
                        "training thread.",
            affects_outcome_of=ModelLifecycle.NONE
        )

        num_checkpoints = configurable_integer(
            default_value=5,
            min_value=1,
            max_value=100,
            header="Number of checkpoints that is done during the single training round",
            description="",
            affects_outcome_of=ModelLifecycle.NONE
        )

    @attrs
    class __Postprocessing(ParameterGroup):
        header = string_attribute("Postprocessing")
        description = header

        result_based_confidence_threshold = configurable_boolean(
            default_value=True,
            header="Result based confidence threshold",
            description="Confidence threshold is derived from the results",
            affects_outcome_of=ModelLifecycle.INFERENCE
        )

        confidence_threshold = configurable_float(
            default_value=0.35,
            min_value=0,
            max_value=1,
            header="Confidence threshold",
            description="This threshold only takes effect if the threshold is not set based on the result.",
            affects_outcome_of=ModelLifecycle.INFERENCE
        )

    @attrs
    class __AlgoBackend(ParameterGroup):
        header = string_attribute("Internal Algo Backend parameters")
        description = header
        visible_in_ui = False

        template = configurable_str("template.yaml", "", editable=False, visible_in_ui=False)
        model = configurable_str("model.py", "", editable=False, visible_in_ui=False)
        model_name = configurable_str("object detection model", "", editable=False, visible_in_ui=False)
        data_pipeline = configurable_str("ote_data_pipeline.py", "", editable=False, visible_in_ui=False)

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
    postprocessing = add_parameter_group(__Postprocessing)
