import io
import numpy as np
import os.path as osp
import random
import time
import torch
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from sc_sdk.entities.annotation import (Annotation, AnnotationScene,
                                        AnnotationSceneKind)
from sc_sdk.entities.dataset_item import DatasetItem
from sc_sdk.entities.datasets import Dataset, NullDatasetStorage, Subset
from sc_sdk.entities.id import ID
from sc_sdk.entities.image import Image
from sc_sdk.entities.inference_parameters import InferenceParameters
from sc_sdk.entities.media_identifier import ImageIdentifier
from sc_sdk.entities.metrics import Performance
from sc_sdk.entities.model import (Model, ModelStatus, NullModel,
                                   NullModelStorage)
from sc_sdk.entities.optimized_model import (ModelOptimizationType,
                                             ModelPrecision, OptimizedModel,
                                             TargetDevice)
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.shapes.ellipse import Ellipse
from sc_sdk.entities.shapes.polygon import Polygon
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.tests.test_helpers import generate_random_annotated_image
from sc_sdk.usecases.tasks.interfaces.export_interface import (ExportType,
                                                               IExportTask)
from sc_sdk.utils import restricted_pickle_module
from sc_sdk.utils.project_factory import NullProject

from mmdet.apis.ote.apis.detection import (OpenVINODetectionTask,
                                           OTEDetectionConfig,
                                           OTEDetectionTask)
from mmdet.apis.ote.apis.detection.config_utils import \
    apply_template_configurable_parameters
from mmdet.apis.ote.apis.detection.ote_utils import (generate_label_schema,
                                                     load_template)


class TestOTEAPI(unittest.TestCase):
    """
    Collection of tests for OTE API and OTE Model Templates
    """

    def init_environment(self, params, number_of_images=500):
        labels_names = ('rectangle', 'ellipse', 'triangle')
        labels_schema = generate_label_schema(labels_names)
        labels_list = labels_schema.get_labels(False)
        environment = TaskEnvironment(model=NullModel(), configurable_parameters=params, label_schema=labels_schema)

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

            image = Image(name=f'image_{i}', project=NullProject(), numpy=image_numpy)
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
        template = load_template(osp.join(template_dir, 'template.yaml'))
        self.assertEqual(template['task']['base'], 'mmdet.apis.ote.apis.detection.OTEDetectionTask')
        self.assertEqual(template['task']['openvino'], 'mmdet.apis.ote.apis.detection.OpenVINODetectionTask')
        self.assertEqual(template['hyper_parameters']['impl'], 'mmdet.apis.ote.apis.detection.OTEDetectionConfig')
        configurable_parameters = OTEDetectionConfig(workspace_id=ID(), model_storage_id=ID())
        apply_template_configurable_parameters(configurable_parameters, template)
        configurable_parameters.learning_parameters.num_iters = num_iters
        configurable_parameters.learning_parameters.num_checkpoints = 1
        configurable_parameters.postprocessing.result_based_confidence_threshold = False
        configurable_parameters.postprocessing.confidence_threshold = 0.1
        return configurable_parameters

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
        configurable_parameters = self.setup_configurable_parameters(template_dir, num_iters=500)
        detection_environment, dataset = self.init_environment(configurable_parameters, 250)
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
        while not detection_task.is_training:
            time.sleep(10)
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        train_future.result()
        self.assertLess(time.time() - start_time, 35, 'Expected to stop within 35 seconds.')

        # Test stopping immediately (as soon as training is started).
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset, output_model)
        while not detection_task.is_training:
            time.sleep(0.1)
        detection_task.cancel_training()

        train_future.result()
        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds

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
        - Trains a model for 10 epochs. Asserts that the returned model is not a NullModel, that
            validation F-measure is larger than the threshold and also that OpenVINO optimization runs successfully.
        - Reloads the model in the task and recompute the performance. Asserts that the performance
            difference between the original and the reloaded model is smaller than 1e-4. Ideally there should be no
            difference at all.
        """
        configurable_parameters = self.setup_configurable_parameters(template_dir, num_iters=150)
        detection_environment, dataset = self.init_environment(configurable_parameters, 250)
        val_dataset = dataset.get_subset(Subset.VALIDATION)
        task = OTEDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print('Task initialized, model training starts.')
        # Train the task.
        # train_task checks that the returned model is not a NullModel, that the task returns an OptimizedModel and that
        # validation f-measure is higher than the threshold, which is a pretty low bar
        # considering that the dataset is so easy
        output_model = Model(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                model_status=ModelStatus.NOT_READY)
        task.train(dataset, output_model)
        self.assertFalse(isinstance(output_model, NullModel))

        # Test that labels and configurable parameters are stored in model.data
        modelinfo = torch.load(io.BytesIO(output_model.get_data("weights.pth")))
                               # pickle_module=restricted_pickle_module)
        self.assertEqual(list(modelinfo.keys()), ['model', 'config', 'labels', 'VERSION'])
        self.assertTrue('ellipse' in modelinfo['labels'])

        if isinstance(task, IExportTask):
            exported_model = OptimizedModel(
                NullProject(),
                NullModelStorage(),
                dataset,
                detection_environment.get_model_configuration(),
                ModelOptimizationType.MO,
                [ModelPrecision.FP32],
                optimization_methods=[],
                optimization_level={},
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
        task.hyperparams.learning_parameters.num_iters = 10
        task.hyperparams.learning_parameters.num_checkpoints = 1
        task.train(dataset, new_model)
        self.assertTrue(first_model.model_status)
        self.assertNotEqual(first_model, new_model)

        # Make the new model fail
        new_model.model_status = ModelStatus.NOT_IMPROVED
        detection_environment.model = first_model
        task = OTEDetectionTask(detection_environment)
        self.assertEqual(task.task_environment.model.id, first_model.id)

        print('Reevaluating model.')
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, output_model, val_dataset)
        performance_delta = performance_after_reloading.score.value - validation_performance.score.value
        perf_delta_tolerance = 0.0005

        self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f'Expected no or very small performance difference after reloading. Performance delta '
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
            self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f'Expected no or very small performance difference after export. Performance delta '
                            f'({validation_performance.score.value} vs {export_performance.score.value}) was '
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

    def test_training_custom_mobilenet_vfnet(self):
        self.train_and_eval(osp.join('configs', 'ote', 'custom-object-detection', 'resnet50_VFNet'))
