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
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import unittest
import warnings
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor
from e2e_test_system import e2e_pytest_api
from ote_sdk.configuration.helper import convert, create
from ote_sdk.entities.annotation import (Annotation, AnnotationSceneEntity,
                                         AnnotationSceneKind)
from ote_sdk.entities.dataset_item import DatasetItemEntity
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.image import Image
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model import (ModelEntity, ModelFormat,
                                    ModelOptimizationType, ModelPrecision,
                                    ModelStatus, OptimizationMethod)
from ote_sdk.entities.model_template import TargetDevice, parse_model_template
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from ote_sdk.tests.test_helpers import generate_random_annotated_image
from ote_sdk.usecases.tasks.interfaces.export_interface import (ExportType,
                                                                IExportTask)
from ote_sdk.usecases.tasks.interfaces.optimization_interface import \
    OptimizationType
from subprocess import run
from typing import Optional

from mmdet.apis.ote.apis.detection import (OpenVINODetectionTask,
                                           OTEDetectionConfig,
                                           OTEDetectionInferenceTask,
                                           OTEDetectionNNCFTask,
                                           OTEDetectionTrainingTask)
from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import generate_label_schema
from mmdet.integration.nncf.utils import is_nncf_enabled

DEFAULT_TEMPLATE_DIR = osp.join('configs', 'ote', 'custom-object-detection',
                                'gen3_mobilenetV2_ATSS')


class ModelTemplate(unittest.TestCase):

    # ------------------------------ Gen1 ------------------------------

    @e2e_pytest_api
    def test_reading_gen1_ssd(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen1_mobilenet_v2-2s_ssd-256x256', 'template.yaml'))

    # ------------------------------ Gen2 ------------------------------

    @e2e_pytest_api
    def test_reading_gen2_ssd(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_mobilenetV2_SSD', 'template.yaml'))

    @e2e_pytest_api
    def test_reading_gen2_atss(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_mobilenetV2_ATSS', 'template.yaml'))

    @e2e_pytest_api
    def test_reading_gen2_vfnet(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_resnet50_VFNet', 'template.yaml'))

    # ------------------------------ Gen3 ------------------------------

    @e2e_pytest_api
    def test_reading_gen3_ssd(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_mobilenetV2_SSD', 'template.yaml'))

    @e2e_pytest_api
    def test_reading_gen3_atss(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_mobilenetV2_ATSS', 'template.yaml'))

    @e2e_pytest_api
    def test_reading_gen3_vfnet(self):
        parse_model_template(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_resnet50_VFNet', 'template.yaml'))


@e2e_pytest_api
def test_configuration_yaml():
    configuration = OTEDetectionConfig()
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = create(configuration_yaml_str)
    configuration_yaml_loaded = create(
        osp.join('mmdet', 'apis', 'ote', 'apis', 'detection',
                 'configuration.yaml'))
    assert configuration_yaml_converted == configuration_yaml_loaded


@e2e_pytest_api
def test_set_values_as_default():
    template_file = osp.join(DEFAULT_TEMPLATE_DIR, 'template.yaml')
    model_template = parse_model_template(template_file)

    hyper_parameters = model_template.hyper_parameters.data
    # value that comes from template.yaml
    default_value = hyper_parameters['learning_parameters']['batch_size'][
        'default_value']
    # value that comes from OTEDetectionConfig
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert value == 5
    assert default_value == 8

    # after this call value must be equal to default_value
    set_values_as_default(hyper_parameters)
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert default_value == value
    hyper_parameters = create(hyper_parameters)
    assert default_value == hyper_parameters.learning_parameters.batch_size


class Sample(unittest.TestCase):
    root_dir = '/tmp'
    coco_dir = osp.join(root_dir, 'data/coco')
    snapshots_dir = osp.join(root_dir, 'snapshots')
    template = osp.join(DEFAULT_TEMPLATE_DIR, 'template.yaml')

    custom_operations = [
        'ExperimentalDetectronROIFeatureExtractor', 'PriorBox',
        'PriorBoxClustered', 'DetectionOutput', 'DeformableConv2D'
    ]

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted(
                [item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [
                item for item in content['images']
                if item['id'] in selected_indexes
            ]
            content['annotations'] = [
                item for item in content['annotations']
                if item['image_id'] in selected_indexes
            ]
            content['licenses'] = [
                item for item in content['licenses']
                if item['id'] in selected_indexes
            ]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
                check=True,
                shell=True)
        if not osp.exists(
                osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
                check=True,
                shell=True)

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 100

        cls.shorten_annotation(
            osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
            osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
            cls.shorten_to)

    @e2e_pytest_api
    def test_sample_on_cpu(self):
        output = run(
            'export CUDA_VISIBLE_DEVICES=;'
            'python mmdet/apis/ote/sample/sample.py '
            f'--data-dir {self.coco_dir}/.. '
            f'--export {self.template}',
            shell=True,
            check=True)
        assert output.returncode == 0

    @e2e_pytest_api
    def test_sample_on_gpu(self):
        output = run(
            'export CUDA_VISIBLE_DEVICES=0;'
            'python mmdet/apis/ote/sample/sample.py '
            f'--data-dir {self.coco_dir}/.. '
            f'--export {self.template}',
            shell=True,
            check=True)
        assert output.returncode == 0


class API(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    def init_environment(self, params, model_template, number_of_images=500):
        labels_names = ('rectangle', 'ellipse', 'triangle')
        labels_schema = generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(
            model=None,
            hyper_parameters=params,
            label_schema=labels_schema,
            model_template=model_template)

        warnings.filterwarnings(
            'ignore', message='.* coordinates .* are out of bounds.*')
        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(
                image_width=640,
                image_height=480,
                labels=labels_list,
                max_shapes=20,
                min_size=50,
                max_size=100,
                random_seed=None)
            # Convert all shapes to bounding boxes
            box_shapes = []
            for shape in shapes:
                shape_labels = shape.get_labels(include_empty=True)
                shape = shape.shape
                if isinstance(shape, (Rectangle, Ellipse)):
                    box = np.array([shape.x1, shape.y1, shape.x2, shape.y2],
                                   dtype=float)
                elif isinstance(shape, Polygon):
                    box = np.array(
                        [shape.min_x, shape.min_y, shape.max_x, shape.max_y],
                        dtype=float)
                box = box.clip(0, 1)
                box_shapes.append(
                    Annotation(
                        Rectangle(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                        labels=shape_labels))

            image = Image(data=image_numpy)
            annotation = AnnotationSceneEntity(
                kind=AnnotationSceneKind.ANNOTATION, annotations=box_shapes)
            items.append(
                DatasetItemEntity(media=image, annotation_scene=annotation))
        warnings.resetwarnings()

        rng = random.Random()
        rng.shuffle(items)
        for i, _ in enumerate(items):
            subset_region = i / number_of_images
            if subset_region >= 0.8:
                subset = Subset.TESTING
            elif subset_region >= 0.6:
                subset = Subset.VALIDATION
            else:
                subset = Subset.TRAINING
            items[i].subset = subset

        dataset = DatasetEntity(items)
        return environment, dataset

    def setup_configurable_parameters(self, template_dir, num_iters=10):
        model_template = parse_model_template(
            osp.join(template_dir, 'template.yaml'))

        hyper_parameters = model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)
        hyper_parameters = create(hyper_parameters)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        return hyper_parameters, model_template

    @e2e_pytest_api
    def test_cancel_training_detection(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Start training and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_TEMPLATE_DIR, num_iters=500)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 64)

        detection_task = OTEDetectionTrainingTask(
            task_environment=detection_environment)

        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='train_thread')

        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)

        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset,
                                       output_model, train_parameters)
        # give train_thread some time to initialize the model
        while not detection_task._is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertEqual(training_progress_curve[-1], 100)
        self.assertLess(time.time() - start_time, 100,
                        'Expected to stop within 100 seconds.')

        # Test stopping immediately (as soon as training is started).
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset,
                                       output_model)
        while not detection_task._is_training:
            time.sleep(0.1)
        detection_task.cancel_training()

        train_future.result()
        self.assertLess(
            time.time() - start_time,
            25)  # stopping process has to happen in less than 25 seconds

    @e2e_pytest_api
    def test_training_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_TEMPLATE_DIR, num_iters=5)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 50)

        task = OTEDetectionTrainingTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model, train_parameters)

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(
            np.all(
                training_progress_curve[1:] >= training_progress_curve[:-1]))

    @e2e_pytest_api
    def test_nncf_optimize_progress_tracking(self):
        if not is_nncf_enabled():
            self.skipTest("Required NNCF module.")

        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_TEMPLATE_DIR, num_iters=2)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 50)

        task = OTEDetectionTrainingTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        original_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task.train(dataset, original_model, TrainParameters)

        # Create NNCFTask
        detection_environment.model = original_model
        nncf_task = OTEDetectionNNCFTask(
            task_environment=detection_environment)
        self.addCleanup(nncf_task._delete_scratch_space)

        # Rewrite some parameters to spend less time
        nncf_task._config["runner"]["max_epochs"] = 10
        nncf_init_cfg = nncf_task._config["nncf_config"]["compression"][0][
            "initializer"]
        nncf_init_cfg["range"]["num_init_samples"] = 1
        nncf_init_cfg["batchnorm_adaptation"]["num_bn_adaptation_samples"] = 1

        print('Task initialized, model optimization starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        optimization_parameters = OptimizationParameters
        optimization_parameters.update_progress = progress_callback
        nncf_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        optimization_parameters.update_progress = progress_callback

        nncf_task.optimize(OptimizationType.NNCF, dataset, nncf_model,
                           optimization_parameters)

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(
            np.all(
                training_progress_curve[1:] >= training_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_progress_tracking(self):
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_TEMPLATE_DIR, num_iters=10)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 50)

        task = OTEDetectionTrainingTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model inference starts.')
        inference_progress_curve = []

        def inference_progress_callback(progress: float,
                                        score: Optional[float] = None):
            inference_progress_curve.append(progress)

        inference_parameters = InferenceParameters
        inference_parameters.update_progress = inference_progress_callback

        task.infer(dataset.with_empty_annotations(), inference_parameters)

        self.assertGreater(len(inference_progress_curve), 0)
        inference_progress_curve = np.asarray(inference_progress_curve)
        self.assertTrue(
            np.all(
                inference_progress_curve[1:] >= inference_progress_curve[:-1]))

    @e2e_pytest_api
    def test_inference_task(self):
        # Prepare pretrained weights
        hyper_parameters, model_template = self.setup_configurable_parameters(
            DEFAULT_TEMPLATE_DIR, num_iters=2)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 50)
        val_dataset = dataset.get_subset(Subset.VALIDATION)

        train_task = OTEDetectionTrainingTask(
            task_environment=detection_environment)
        self.addCleanup(train_task._delete_scratch_space)

        trained_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        train_task.train(dataset, trained_model, TrainParameters)
        performance_after_train = self.eval(train_task, trained_model,
                                            val_dataset)

        # Create InferenceTask
        detection_environment.model = trained_model
        inference_task = OTEDetectionInferenceTask(
            task_environment=detection_environment)
        self.addCleanup(inference_task._delete_scratch_space)

        performance_after_load = self.eval(inference_task, trained_model,
                                           val_dataset)

        assert performance_after_train == performance_after_load

        # Export
        exported_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
            _id=ObjectId())
        inference_task.export(ExportType.OPENVINO, exported_model)

    @staticmethod
    def eval(task: OTEDetectionTrainingTask, model: ModelEntity,
             dataset: DatasetEntity) -> Performance:
        start_time = time.time()
        result_dataset = task.infer(dataset.with_empty_annotations())
        end_time = time.time()
        print(f'{len(dataset)} analysed in {end_time - start_time} seconds')
        result_set = ResultSetEntity(
            model=model,
            ground_truth_dataset=dataset,
            prediction_dataset=result_dataset)
        task.evaluate(result_set)
        assert result_set.performance is not None
        return result_set.performance

    def check_threshold(self, reference, value, delta_tolerance, message=''):
        delta = value.score.value - reference.score.value
        self.assertLessEqual(
            np.abs(delta),
            delta_tolerance,
            msg=message + f' (reference metric: {reference.score.value}, '
            f'actual value: {value.score.value}, '
            f'delta tolerance threshold: {delta_tolerance})')

    def end_to_end(self,
                   template_dir,
                   num_iters=5,
                   quality_score_threshold=0.5,
                   reload_perf_delta_tolerance=0.0,
                   export_perf_delta_tolerance=0.0005,
                   pot_perf_delta_tolerance=0.1,
                   nncf_perf_delta_tolerance=0.1):

        hyper_parameters, model_template = self.setup_configurable_parameters(
            template_dir, num_iters=num_iters)
        detection_environment, dataset = self.init_environment(
            hyper_parameters, model_template, 250)

        val_dataset = dataset.get_subset(Subset.VALIDATION)
        task = OTEDetectionTrainingTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        # Train the task.
        # train_task checks that the task returns an Model and that
        # validation f-measure is higher than the threshold, which is a pretty low bar
        # considering that the dataset is so easy
        output_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
            _id=ObjectId())
        task.train(dataset, output_model)

        # Test that output model is valid.
        self.assertEqual(output_model.model_status, ModelStatus.SUCCESS)
        modelinfo = torch.load(
            io.BytesIO(output_model.get_data("weights.pth")))
        self.assertEqual(
            list(modelinfo.keys()), ['model', 'config', 'labels', 'VERSION'])
        self.assertTrue('ellipse' in modelinfo['labels'])

        # Run inference.
        validation_performance = self.eval(task, output_model, val_dataset)
        print(f'Performance: {validation_performance.score.value:.4f}')
        self.assertGreater(
            validation_performance.score.value, quality_score_threshold,
            f'Expected F-measure to be higher than {quality_score_threshold}')

        # Run another training round.
        first_model = output_model
        new_model = ModelEntity(
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY,
            _id=ObjectId())
        task._hyperparams.learning_parameters.num_iters = 1
        task._hyperparams.learning_parameters.num_checkpoints = 1
        task.train(dataset, new_model)
        self.assertEqual(new_model.model_status, ModelStatus.SUCCESS)
        self.assertNotEqual(first_model, new_model)
        self.assertNotEqual(
            first_model.get_data("weights.pth"),
            new_model.get_data("weights.pth"))

        # Make the new model fail.
        new_model.model_status = ModelStatus.NOT_IMPROVED
        detection_environment.model = first_model
        task = OTEDetectionTrainingTask(detection_environment)
        self.assertEqual(task._task_environment.model.id, first_model.id)

        print('Reevaluating model.')
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, output_model,
                                                val_dataset)
        print(
            f'Performance after reloading: {performance_after_reloading.score.value:.4f}'
        )
        self.check_threshold(
            validation_performance, performance_after_reloading,
            reload_perf_delta_tolerance,
            'Too big performance difference after model reload.')

        if isinstance(task, IExportTask):
            # Run export.
            exported_model = ModelEntity(
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY,
                _id=ObjectId())
            task.export(ExportType.OPENVINO, exported_model)
            self.assertEqual(exported_model.model_status, ModelStatus.SUCCESS)
            self.assertEqual(exported_model.model_format, ModelFormat.OPENVINO)
            self.assertEqual(exported_model.optimization_type,
                             ModelOptimizationType.MO)

            # Create OpenVINO Task and evaluate the model.
            detection_environment.model = exported_model
            ov_task = OpenVINODetectionTask(detection_environment)
            predicted_validation_dataset = ov_task.infer(
                val_dataset.with_empty_annotations())
            resultset = ResultSetEntity(
                model=output_model,
                ground_truth_dataset=val_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            ov_task.evaluate(resultset)
            export_performance = resultset.performance
            assert export_performance is not None
            print(
                f'Performance of exported model: {export_performance.score.value:.4f}'
            )
            self.check_threshold(
                validation_performance, export_performance,
                export_perf_delta_tolerance,
                'Too big performance difference after OpenVINO export.')

            # Run POT optimization and evaluate the result.
            print('Run POT optimization.')
            optimized_model = ModelEntity(
                dataset,
                detection_environment.get_model_configuration(),
                optimization_type=ModelOptimizationType.POT,
                optimization_methods=OptimizationMethod.QUANTIZATION,
                optimization_objectives={},
                precision=[ModelPrecision.INT8],
                target_device=TargetDevice.CPU,
                performance_improvement={},
                model_size_reduction=1.,
                model_status=ModelStatus.NOT_READY)
            ov_task.optimize(OptimizationType.POT, dataset, optimized_model,
                             OptimizationParameters())
            pot_performance = self.eval(ov_task, optimized_model, val_dataset)
            print(
                f'Performance of optimized model: {pot_performance.score.value:.4f}'
            )
            self.check_threshold(
                validation_performance, pot_performance,
                pot_perf_delta_tolerance,
                'Too big performance difference after POT optimization.')

        if model_template.entrypoints.nncf:
            if is_nncf_enabled():
                print('Run NNCF optimization.')
                nncf_model = ModelEntity(
                    dataset,
                    detection_environment.get_model_configuration(),
                    optimization_type=ModelOptimizationType.NNCF,
                    optimization_methods=OptimizationMethod.QUANTIZATION,
                    optimization_objectives={},
                    precision=[ModelPrecision.INT8],
                    target_device=TargetDevice.CPU,
                    performance_improvement={},
                    model_size_reduction=1.,
                    model_status=ModelStatus.NOT_READY)
                nncf_model.set_data('weights.pth',
                                    output_model.get_data("weights.pth"))

                detection_environment.model = nncf_model

                nncf_task = OTEDetectionNNCFTask(
                    task_environment=detection_environment)

                nncf_task.optimize(OptimizationType.NNCF, dataset, nncf_model,
                                   OptimizationParameters())
                nncf_task.save_model(nncf_model)
                nncf_performance = self.eval(nncf_task, nncf_model,
                                             val_dataset)

                print(
                    f'Performance of NNCF model: {nncf_performance.score.value:.4f}'
                )
                self.check_threshold(
                    validation_performance, nncf_performance,
                    nncf_perf_delta_tolerance,
                    'Too big performance difference after NNCF optimization.')
            else:
                print(
                    'Skipped test of OTEDetectionNNCFTask. Required NNCF module.'
                )

    # ------------------------------ Gen1 ------------------------------

    @e2e_pytest_api
    def test_training_gen1_ssd(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen1_mobilenet_v2-2s_ssd-256x256'),
            num_iters=150)

    # ------------------------------ Gen2 ------------------------------

    @e2e_pytest_api
    def test_training_gen2_ssd(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_mobilenetV2_SSD'),
            num_iters=150)

    @e2e_pytest_api
    def test_training_gen2_atss(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_mobilenetV2_ATSS'),
            num_iters=150)

    @e2e_pytest_api
    def test_training_gen2_vfnet(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen2_resnet50_VFNet'),
            num_iters=150,
            export_perf_delta_tolerance=0.01)

    # ------------------------------ Gen3 ------------------------------

    @e2e_pytest_api
    def test_training_gen3_ssd(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_mobilenetV2_SSD'))

    @e2e_pytest_api
    def test_training_gen3_atss(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_mobilenetV2_ATSS'))

    @e2e_pytest_api
    def test_training_gen3_vfnet(self):
        self.end_to_end(
            osp.join('configs', 'ote', 'custom-object-detection',
                     'gen3_resnet50_VFNet'),
            export_perf_delta_tolerance=0.01)
