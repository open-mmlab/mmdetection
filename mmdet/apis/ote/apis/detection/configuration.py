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

from attr import attrs

from sys import maxsize

from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            boolean_attribute,
                                            configurable_boolean,
                                            configurable_float,
                                            configurable_integer,
                                            selectable,
                                            string_attribute,
                                            ModelConfig)
from ote_sdk.configuration.model_lifecycle import ModelLifecycle

from .configuration_enums import POTQuantizationPreset


@attrs
class OTEDetectionConfig(ModelConfig):
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
        visible_in_ui = boolean_attribute(False)

        template = string_attribute("template.yaml")
        model = string_attribute("model.py")
        model_name = string_attribute("object detection model")
        data_pipeline = string_attribute("ote_data_pipeline.py")

    @attrs
    class __NNCFOptimization(ParameterGroup):
        header = string_attribute("Optimization by NNCF")
        description = header

        config = string_attribute(
            value="compression_config.json"
        )

        apply_quantization = configurable_boolean(
            default_value=False,
            header="Apply quantization by NNCF",
            description="Enable quantization algorithm to get compressed model by NNCF",
            affects_outcome_of=ModelLifecycle.TRAINING
        )
        apply_pruning = configurable_boolean(
            default_value=False,
            header="Apply pruning by NNCF",
            description="Enable pruning algorithm to get compressed model by NNCF",
            affects_outcome_of=ModelLifecycle.TRAINING
        )
        apply_pruning_quantization = configurable_boolean(
            default_value=False,
            header="Apply pruning and quantization by NNCF",
            description="Enable pruning and quantization algorithm to get compressed model by NNCF",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    class __POTParameter(ParameterGroup):
        header = string_attribute("POT Parameters")
        description = header

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize
        )

        preset = selectable(default_value=POTQuantizationPreset.PERFORMANCE, header="Preset",
                            description="Quantization preset that defines quantization scheme",
                            editable=False, visible_in_ui=False)

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
    postprocessing = add_parameter_group(__Postprocessing)
    nncf_optimization = add_parameter_group(__NNCFOptimization)
    pot_parameters = add_parameter_group(__POTParameter)
