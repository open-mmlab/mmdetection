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

import os
import os.path as osp

from sc_sdk.configuration.configurable_parameters import *
from sc_sdk.configuration.deep_learning_configurable_parameters import DeepLearningConfigurableParameters


def list_available_models(models_directory):
    available_models = []
    for dirpath, dirnames, filenames in os.walk(models_directory):
        for filename in filenames:
            if filename == 'model.py':
                available_models.append(dict(
                    name=osp.basename(dirpath),
                    dir=osp.join(models_directory, dirpath)))
    available_models.sort(key=lambda x: x['name'])
    return available_models


class MMDetectionParameters(DeepLearningConfigurableParameters):

    class __Postprocessing(Group):
        header = "Postprocessing"
        description = header

        cross_class_nms = Boolean(
            header="Cross class non-maximum suppression",
            default_value=False
        )

        result_based_confidence_threshold = Boolean(
            header="Result based confidence threshold",
            description="Confidence threshold is derived from the results",
            default_value=True
        )

        confidence_threshold = Float(
            header="Confidence threshold",
            description="This threshold only takes effect if the threshold is not set based on the result.",
            default_value=0.35,
            min_value=0,
            max_value=1
        )

        result_based_nms_threshold = Boolean(
            header="Result based NMS threshold",
            description="The NMS threshold is derived from the results",
            default_value=False
        )

        nms_threshold = Float(
            header="NMS threshold",
            description="The NMS threshold used to match the detections and ground truths",
            default_value=0.5,
            min_value=0.1,
            max_value=1
        )

    class __NOUSParameters(Group):
        header = "NOUS Active Learning Parameters"
        description = header

        auto_training = Boolean(
            header="Auto-training",
            description="Set to True to allow NOUS start training automatically when it is ready "
            "to train. In a pipeline, the auto-training parameter for the first "
            "deep learning task will be considered.",
            default_value=True,
        )

        max_unseen_media = Integer(
            header="Number of images analysed after training for active learning",
            description="For pipeline projects, the number of images for the first task is used "
            "to generate results for the pipeline",
            default_value=250,
            min_value=10,
            max_value=10000,
        )

    class __LearningParameters(Group):
        header = "Learning Parameters"
        description = header

        batch_size = Integer(
            default_value=5,
            min_value=1,
            max_value=32,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "reduces training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            editable=True,
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
        )

        num_epochs = Integer(
            default_value=1,
            header="Number of epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            min_value=1,
            max_value=1000,
            editable=True,
        )

        learning_rate = Float(
            default_value=0.0025,
            header="Learning rate",
            description="Increasing this value will speed up training convergence but might make it unstable. A value "
                        "of 0.0025 is recommended.",
            min_value=1e-06,
            max_value=1e-02,
            editable=True,
        )

        learning_rate_warmup_iters = Integer(
            default_value=100,
            header="Number of iterations for learning rate warmup",
            description="",
            min_value=0,
            max_value=1000,
            editable=True,
        )

        learning_rate_schedule = Selectable(header="Learning rate schedule",
                                            default_value="exp",
                                            options=[Option(key="fixed",
                                                            value="Fixed",
                                                            description="Learning rate is kept fixed during training"),
                                                     Option(key="exp",
                                                            value="Exponential annealing",
                                                            description="Learning rate is reduced exponentially"),
                                                     Option(key="step",
                                                            value="Step-wise annealing",
                                                            description="Learning rate is reduced step-wise at "
                                                                        "epoch 10"),
                                                     Option(key="cyclic",
                                                            value="Cyclic cosine annealing",
                                                            description="Learning rate is gradually reduced and "
                                                                        "increased during training, following a cosine "
                                                                        "pattern. The pattern is repeated for two "
                                                                        "cycles in one training run."),
                                                     Option(key="custom",
                                                            value="Custom",
                                                            description="Learning rate schedule that is provided "
                                                                        "with the model."),
                                                     ],
                                            description="Specify learning rate scheduling for the MMDetection task. "
                                                        "When training for a small number of epochs (N < 10), the fixed"
                                                        " schedule is recommended. For training for 10 < N < 25 epochs,"
                                                        " step-wise or exponential annealing might give better results."
                                                        " Finally, for training on large datasets for at least 20 "
                                                        "epochs, cyclic annealing could result in the best model.",
                                            editable=True)

    class __AlgoBackend(Group):
        header = "Internal Algo Backend parameters"
        description = header
        base_models_dir = osp.join(osp.abspath(osp.dirname(__file__)), '..', '..', '..', 'configs', 'ote', 'custom-object-detection')
        model_name = Selectable(header="Model name",
                                  default_value='model',
                                  options=[Option(key='model', value='model',
                                                  description='Path to model configuration file')],
                                  description="Model name.",
                                  editable=False)
        model = Selectable(header="Model architecture",
                                  default_value='model.py',
                                  options=[Option(key='model.py', value='model.py',
                                                  description='Path to model configuration file')],
                                  description="Specify learning architecture for the the task.",
                                  editable=False)
        data_pipeline = Selectable(header="Data pipeline",
                                   default_value='nous_data_pipeline.py',
                                   options=[Option(key='nous_data_pipeline.py', value='nous_data_pipeline.py',
                                                   description='Path to data pipeline configuration file')],
                                   description="Specify data pipeline for the the task.",
                                   editable=False)

    # class __LearningArchitecture(Group):
    #     header = "Learning Architecture"
    #     description = header
    #     base_models_dir = osp.join(osp.abspath(osp.dirname(__file__)), '..', '..', '..', 'configs', 'ote', 'custom-object-detection')
    #     available_models = list_available_models(base_models_dir)
    #     model_architecture = Selectable(header="Model architecture",
    #                                     default_value=available_models[0]['name'],
    #                                     options=[Option(key=x['name'], value=x['name'], description='') for x in available_models],
    #                                     description="Specify learning architecture for the the task.",
    #                                     editable=True)

    learning_parameters: __LearningParameters = Object(__LearningParameters)
    nous_parameters: __NOUSParameters = Object(__NOUSParameters)
    postprocessing: __Postprocessing = Object(__Postprocessing)
    algo_backend: __AlgoBackend = Object(__AlgoBackend)

