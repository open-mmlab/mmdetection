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
import yaml
from concurrent.futures import ThreadPoolExecutor
from ote_sdk.configuration.helper import convert, create
from ote_sdk.entities.annotation import Annotation, AnnotationSceneKind
from ote_sdk.entities.id import ID
from ote_sdk.entities.metrics import Performance
from ote_sdk.entities.model_template import parse_model_template, TargetDevice
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.shapes.box import Box
from ote_sdk.entities.shapes.ellipse import Ellipse
from ote_sdk.entities.shapes.polygon import Polygon
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.train_parameters import TrainParameters
from sc_sdk.entities.annotation import AnnotationScene
from sc_sdk.entities.dataset_item import DatasetItem
from sc_sdk.entities.datasets import Dataset, NullDatasetStorage, Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.media_identifier import ImageIdentifier
from sc_sdk.entities.model import Model, ModelStatus, NullModelStorage
from ote_sdk.entities.model import (ModelPrecision,
                                    ModelStatus,
                                    ModelOptimizationType,
                                    OptimizationMethod)
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.tests.test_helpers import generate_random_annotated_image
from ote_sdk.usecases.tasks.interfaces.export_interface import (ExportType,
                                                               IExportTask)
from ote_sdk.usecases.tasks.interfaces.optimization_interface import OptimizationType
from sc_sdk.utils.project_factory import NullProject
from subprocess import run
from typing import Optional

from mmdet.apis.ote.apis.detection import (OpenVINODetectionTask,
                                           OTEDetectionConfig,
                                           OTEDetectionTask)
from mmdet.apis.ote.apis.detection.config_utils import set_values_as_default
from mmdet.apis.ote.apis.detection.ote_utils import generate_label_schema


class ModelTemplate(unittest.TestCase):

    def test_reading_mnv2_ssd_256(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml')

    def test_reading_mnv2_ssd_384(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-384x384/template.yaml')

    def test_reading_mnv2_ssd_512(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-512x512/template.yaml')

    def test_reading_mnv2_ssd(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenetV2_SSD/template.yaml')

    def test_reading_mnv2_atss(self):
        parse_model_template('./configs/ote/custom-object-detection/mobilenetV2_ATSS/template.yaml')

    def test_reading_resnet50_vfnet(self):
        parse_model_template('./configs/ote/custom-object-detection/resnet50_VFNet/template.yaml')

def test_configuration_yaml():
    configuration = OTEDetectionConfig(workspace_id=ID(), model_storage_id=ID())
    configuration_yaml_str = convert(configuration, str)
    configuration_yaml_converted = yaml.safe_load(configuration_yaml_str)
    with open(osp.join('mmdet', 'apis', 'ote', 'apis', 'detection', 'configuration.yaml')) as read_file:
        configuration_yaml_loaded = yaml.safe_load(read_file)
    del configuration_yaml_converted['algo_backend']
    assert configuration_yaml_converted == configuration_yaml_loaded

def test_set_values_as_default():
    template_dir = './configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/'
    template_file = osp.join(template_dir, 'template.yaml')
    model_template = parse_model_template(template_file)

    hyper_parameters = model_template.hyper_parameters.data
    # value that comes from template.yaml
    default_value = hyper_parameters['learning_parameters']['batch_size']['default_value']
    # value that comes from OTEDetectionConfig
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert value == 5
    assert default_value == 64

    # after this call value must be equal to default_value
    set_values_as_default(hyper_parameters)
    value = hyper_parameters['learning_parameters']['batch_size']['value']
    assert default_value == value
    hyper_parameters = create(hyper_parameters)
    assert default_value == hyper_parameters.learning_parameters.batch_size

class SampleTestCase(unittest.TestCase):
    root_dir = '/tmp'
    coco_dir = osp.join(root_dir, 'data/coco')
    snapshots_dir = osp.join(root_dir, 'snapshots')

    custom_operations = ['ExperimentalDetectronROIFeatureExtractor',
                         'PriorBox', 'PriorBoxClustered', 'DetectionOutput',
                         'DeformableConv2D']

    @staticmethod
    def shorten_annotation(src_path, dst_path, num_images):
        with open(src_path) as read_file:
            content = json.load(read_file)
            selected_indexes = sorted([item['id'] for item in content['images']])
            selected_indexes = selected_indexes[:num_images]
            content['images'] = [item for item in content['images'] if
                                 item['id'] in selected_indexes]
            content['annotations'] = [item for item in content['annotations'] if
                                      item['image_id'] in selected_indexes]
            content['licenses'] = [item for item in content['licenses'] if
                                   item['id'] in selected_indexes]

        with open(dst_path, 'w') as write_file:
            json.dump(content, write_file)

    @classmethod
    def setUpClass(cls):
        cls.test_on_full = False
        os.makedirs(cls.coco_dir, exist_ok=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017.zip')):
            run(f'wget --no-verbose http://images.cocodataset.org/zips/val2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'val2017')):
            run(f'unzip {osp.join(cls.coco_dir, "val2017.zip")} -d {cls.coco_dir}', check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, "annotations_trainval2017.zip")):
            run(f'wget --no-verbose http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {cls.coco_dir}',
            check=True, shell=True)
        if not osp.exists(osp.join(cls.coco_dir, 'annotations/instances_val2017.json')):
            run(f'unzip -o {osp.join(cls.coco_dir, "annotations_trainval2017.zip")} -d {cls.coco_dir}',
            check=True, shell=True)

        if cls.test_on_full:
            cls.shorten_to = 5000
        else:
            cls.shorten_to = 100

        cls.shorten_annotation(osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               osp.join(cls.coco_dir, 'annotations/instances_val2017.json'),
                               cls.shorten_to)

    def test_sample_on_cpu(self):
        output = run('export CUDA_VISIBLE_DEVICES=;'
                     'python mmdet/apis/ote/sample/sample.py '
                     f'--data-dir {self.coco_dir}/.. '
                     '--export configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml',
                     shell=True, check=True)
        assert output.returncode == 0

    def test_sample_on_gpu(self):
        output = run('export CUDA_VISIBLE_DEVICES=0;'
                     'python mmdet/apis/ote/sample/sample.py '
                     f'--data-dir {self.coco_dir}/.. '
                     '--export configs/ote/custom-object-detection/mobilenet_v2-2s_ssd-256x256/template.yaml',
                     shell=True, check=True)
        assert output.returncode == 0


class TestOTEAPI(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    def init_environment(self, params, model_template, number_of_images=500):
        labels_names = ('rectangle', 'ellipse', 'triangle')
        labels_schema = generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(model=None, hyper_parameters=params, label_schema=labels_schema,
                                      model_template=model_template)

        warnings.filterwarnings('ignore', message='.* coordinates .* are out of bounds.*')
        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(image_width=640,
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
                if isinstance(shape, (Box, Ellipse)):
                    box = np.array([shape.x1, shape.y1, shape.x2, shape.y2], dtype=float)
                elif isinstance(shape, Polygon):
                    box = np.array([shape.min_x, shape.min_y, shape.max_x, shape.max_y], dtype=float)
                box = box.clip(0, 1)
                box_shapes.append(Annotation(Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                                             labels=shape_labels))

            image = Image(name=f'image_{i}', numpy=image_numpy, dataset_storage=NullDatasetStorage())
            image_identifier = ImageIdentifier(image.id)
            annotation = AnnotationScene(
                kind=AnnotationSceneKind.ANNOTATION,
                media_identifier=image_identifier,
                annotations=box_shapes)
            items.append(DatasetItem(media=image, annotation_scene=annotation))
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

        dataset = Dataset(NullDatasetStorage(), items)
        return environment, dataset

    def setup_configurable_parameters(self, template_dir, num_iters=250):
        model_template = parse_model_template(osp.join(template_dir, 'template.yaml'))

        hyper_parameters = model_template.hyper_parameters.data
        set_values_as_default(hyper_parameters)
        hyper_parameters = create(hyper_parameters)
        hyper_parameters.learning_parameters.num_iters = num_iters
        hyper_parameters.learning_parameters.num_checkpoints = 1
        hyper_parameters.postprocessing.result_based_confidence_threshold = False
        hyper_parameters.postprocessing.confidence_threshold = 0.1
        return hyper_parameters, model_template

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
        template_dir = osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS')
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=500)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 250)

        detection_task = OTEDetectionTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='train_thread')

        output_model = Model(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        # give train_thread some time to initialize the model
        while not detection_task._is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertLess(time.time() - start_time, 35, 'Expected to stop within 35 seconds.')

        # Test stopping immediately (as soon as training is started).
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        while not detection_task._is_training:
            time.sleep(0.1)
        detection_task.cancel_training()

        train_future.result()
        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

    def test_training_progress_tracking(self):
        template_dir = osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS')
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=10)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 50)

        task = OTEDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        training_progress_curve = []

        def progress_callback(progress: float, score: Optional[float] = None):
            training_progress_curve.append(progress)

        train_parameters = TrainParameters
        train_parameters.update_progress = progress_callback
        output_model = Model(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model, train_parameters)

        self.assertGreater(len(training_progress_curve), 0)
        training_progress_curve = np.asarray(training_progress_curve)
        self.assertTrue(np.all(training_progress_curve[1:] >= training_progress_curve[:-1]))

    @staticmethod
    def eval(task: OTEDetectionTask, model: Model, dataset: Dataset) -> Performance:
        start_time = time.time()
        result_dataset = task.infer(dataset.with_empty_annotations())
        end_time = time.time()
        print(f'{len(dataset)} analysed in {end_time - start_time} seconds')
        result_set = ResultSet(
            model=model,
            ground_truth_dataset=dataset,
            prediction_dataset=result_dataset
        )
        performance = task.evaluate(result_set)
        return performance

    def train_and_eval(self, template_dir):
        """
        Run training, analysis, evaluation and model optimization

        Flow of the test:
        - Creates a randomly annotated project with a small dataset containing 3 classes:
            ['rectangle', 'triangle', 'circle'].
        - Trains a model for 10 epochs. Asserts that validation F-measure is larger than the threshold and
            also that OpenVINO optimization runs successfully.
        - Reloads the model in the task and recompute the performance. Asserts that the performance
            difference between the original and the reloaded model is smaller than 1e-4. Ideally there should be no
            difference at all.
        """
        hyper_parameters, model_template = self.setup_configurable_parameters(template_dir, num_iters=150)
        detection_environment, dataset = self.init_environment(hyper_parameters, model_template, 250)

        val_dataset = dataset.get_subset(Subset.VALIDATION)
        task = OTEDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        # Train the task.
        # train_task checks that the task returns an Model and that
        # validation f-measure is higher than the threshold, which is a pretty low bar
        # considering that the dataset is so easy
        output_model = Model(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model)

        # Test that labels and configurable parameters are stored in model.data
        modelinfo = torch.load(io.BytesIO(output_model.get_data("weights.pth")))
        self.assertEqual(list(modelinfo.keys()), ['model', 'config', 'labels', 'VERSION'])
        self.assertTrue('ellipse' in modelinfo['labels'])

        if isinstance(task, IExportTask):
            exported_model = Model(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                optimization_type=ModelOptimizationType.MO,
                precision=[ModelPrecision.FP32],
                optimization_methods=[],
                optimization_objectives={},
                target_device=TargetDevice.UNSPECIFIED,
                performance_improvement={},
                model_size_reduction=1.,
                model_status=ModelStatus.NOT_READY)
            task.export(ExportType.OPENVINO, exported_model)

        # Run inference
        validation_performance = self.eval(task, output_model, val_dataset)
        print(f'Evaluated model to have a performance of {validation_performance}')
        score_threshold = 0.5
        self.assertGreater(validation_performance.score.value, score_threshold,
            f'Expected F-measure to be higher than {score_threshold}')

        print('Reloading model.')
        first_model = output_model
        new_model = Model(
            NullProject(),
            NullModelStorage(),
            dataset,
            detection_environment.get_model_configuration(),
            model_status=ModelStatus.NOT_READY)
        task._hyperparams.learning_parameters.num_iters = 10
        task._hyperparams.learning_parameters.num_checkpoints = 1
        task.train(dataset, new_model)
        self.assertTrue(first_model.model_status)
        self.assertNotEqual(first_model, new_model)

        # Make the new model fail
        new_model.model_status = ModelStatus.NOT_IMPROVED
        detection_environment.model = first_model
        task = OTEDetectionTask(detection_environment)
        self.assertEqual(task._task_environment.model.id, first_model.id)

        print('Reevaluating model.')
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, output_model, val_dataset)
        performance_delta = performance_after_reloading.score.value - validation_performance.score.value
        perf_delta_tolerance = 0.0

        self.assertEqual(np.abs(performance_delta), perf_delta_tolerance,
                         msg=f'Expected no performance difference after reloading. Performance delta '
                             f'({validation_performance.score.value} vs {performance_after_reloading.score.value}) was '
                             f'larger than the tolerance of {perf_delta_tolerance}')

        print(f'Performance: {validation_performance.score.value:.4f}')
        print(f'Performance after reloading: {performance_after_reloading.score.value:.4f}')
        print(f'Performance delta after reloading: {performance_delta:.6f}')

        if isinstance(task, IExportTask):
            detection_environment.model = exported_model
            ov_task = OpenVINODetectionTask(detection_environment)
            predicted_validation_dataset = ov_task.infer(val_dataset.with_empty_annotations())
            resultset = ResultSet(
                model=output_model,
                ground_truth_dataset=val_dataset,
                prediction_dataset=predicted_validation_dataset,
            )
            export_performance = ov_task.evaluate(resultset)
            print(export_performance)
            performance_delta = export_performance.score.value - validation_performance.score.value
            perf_delta_tolerance = 0.0005
            self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f'Expected no or very small performance difference after export. Performance delta '
                            f'({validation_performance.score.value} vs {export_performance.score.value}) was '
                            f'larger than the tolerance of {perf_delta_tolerance}')

            print('Run POT optimization.')
            optimized_model = Model(
                NullProject(),
                NullModelStorage(),
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
            ov_task.optimize(OptimizationType.POT, dataset, optimized_model, OptimizationParameters())

            pot_performance = self.eval(ov_task, optimized_model, val_dataset)
            print(f'Performance of optimized model: {pot_performance.score.value:.4f}')

            performance_delta = pot_performance.score.value - export_performance.score.value
            perf_delta_tolerance = 0.01
            self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f'Expected not more than one percent performance difference after pot optimization. Performance delta '
                            f'({export_performance.score.value} vs {pot_performance.score.value}) was '
                            f'larger than the tolerance of {perf_delta_tolerance}')

    def test_training_custom_mobilenetssd_256(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-256x256'))

    def test_training_custom_mobilenetssd_384(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-384x384'))

    def test_training_custom_mobilenetssd_512(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-512x512'))

    def test_training_custom_mobilenet_atss(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_ATSS'))

    def test_training_custom_mobilenet_ssd(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'mobilenetV2_SSD'))

    def test_training_custom_resnet_vfnet(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'resnet50_VFNet'))
