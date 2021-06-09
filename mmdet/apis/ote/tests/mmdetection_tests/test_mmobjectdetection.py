import numpy as np
import os.path as osp
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from flaky import flaky
from sc_sdk.entities.annotation import Annotation, AnnotationKind
from sc_sdk.entities.dataset_item import DatasetItem
from sc_sdk.entities.datasets import Dataset, Subset
from sc_sdk.entities.image import Image
from sc_sdk.entities.media_identifier import ImageIdentifier
from sc_sdk.entities.model import NullModel
from sc_sdk.entities.optimized_model import OptimizedModel
from sc_sdk.entities.resultset import ResultSet
from sc_sdk.entities.shapes.box import Box
from sc_sdk.entities.shapes.ellipse import Ellipse
from sc_sdk.entities.shapes.polygon import Polygon
from sc_sdk.entities.task_environment import TaskEnvironment
from sc_sdk.tests.test_helpers import generate_random_annotated_image, rerun_on_flaky_assert
from sc_sdk.usecases.tasks.interfaces.model_optimizer import IModelOptimizer
from sc_sdk.utils.project_factory import ProjectFactory

from mmdet.apis.ote.apis.detection import MMObjectDetectionTask, MMDetectionParameters, configurable_parameters


class TestOTEDetection(unittest.TestCase):
    """
    Collection of OTEDetection tests for the MMObjectDetectionTask
    """

    def init_environment(self, configurable_parameters, number_of_images=500):
        project = ProjectFactory.create_project_single_task(name='OTEDetectionTestProject',
                                                            description='OTEDetectionTestProject',
                                                            label_names=["rectangle", "ellipse", "triangle"],
                                                            task_name="OTEDetectionTestTask",
                                                            configurable_parameters=configurable_parameters)
        self.addCleanup(lambda: ProjectFactory.delete_project_with_id(project.id))
        labels = project.get_labels()

        items = []
        for i in range(0, number_of_images):
            image_numpy, shapes = generate_random_annotated_image(image_width=640,
                                                                  image_height=480,
                                                                  labels=labels,
                                                                  max_shapes=20,
                                                                  min_size=50,
                                                                  max_size=100,
                                                                  random_seed=None)
            # Convert all shapes to bounding boxes
            box_shapes = []
            for shape in shapes:
                shape_labels = shape.get_labels(include_empty=True)
                if isinstance(shape, (Box, Ellipse)):
                    box = np.array([shape.x1, shape.y1, shape.x2, shape.y2], dtype=float)
                elif isinstance(shape, Polygon):
                    box = np.array([shape.min_x, shape.min_y, shape.max_x, shape.max_y], dtype=float)
                box = box.clip(0, 1)
                box_shapes.append(Box(x1=box[0], y1=box[1], x2=box[2], y2=box[3], labels=shape_labels))

            image = Image(name=f"image_{i}", project=project, numpy=image_numpy)
            image_identifier = ImageIdentifier(image.id)
            annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image_identifier)
            annotation.append_shapes(box_shapes)
            items.append(DatasetItem(media=image, annotation=annotation))

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

        dataset = Dataset(items)
        task_node = project.tasks[-1]
        environment = TaskEnvironment(project=project, task_node=task_node)
        return project, environment, dataset

    @staticmethod
    def setup_configurable_parameters(num_epochs=10):
        configurable_parameters = MMDetectionParameters()
        template_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', '..', '..', '..',
                                'configs', 'ote', 'custom-object-detection', 'mobilenet_v2-2s_ssd-256x256')
        configurable_parameters.algo_backend.model.value = osp.join(template_dir, 'model.py')
        configurable_parameters.algo_backend.template.value = osp.join(template_dir, 'template.yaml')
        configurable_parameters.algo_backend.model_name.value = 'some_detection_model'
        configurable_parameters.learning_parameters.learning_rate_schedule.value = 'cyclic'
        configurable_parameters.learning_parameters.batch_size.value = 32
        configurable_parameters.learning_parameters.num_epochs.value = num_epochs
        return configurable_parameters

    @flaky(max_runs=2, rerun_filter=rerun_on_flaky_assert())
    def test_cancel_training_detection(self):
        """
        Tests starting and cancelling training.

        Flow of the test:
        - Creates a randomly annotated project with 100 images and a video with 25 frames. Shapes in the project are
            ['rectangle', 'triangle', 'circle']. Annotations are converted to bounding box annotations.
        - Start training the mmdetection task and give cancel training signal after 10 seconds. Assert that training
            stops within 35 seconds after that
        - Start training the task and give cancel signal immediately. Assert that training stops within 25 seconds.

        This test should be finished in under one minute on a workstation.
        """
        configurable_parameters = self.setup_configurable_parameters(num_epochs=100)
        _, detection_environment, dataset = self.init_environment(configurable_parameters, 250)
        detection_task = MMObjectDetectionTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_thread")

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset)
        time.sleep(10)  # give train_thread some time to initialize the model
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        self.assertLess(time.time() - start_time, 35, "Expected to stop within 35 seconds [flaky].")
        train_future.result()

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(detection_task.train, dataset)
        time.sleep(1.0)
        detection_task.cancel_training()

        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds
        train_future.result()

    @staticmethod
    def eval(task, environment, dataset):
        start_time = time.time()
        result_dataset = task.analyse(dataset.with_empty_annotations())
        end_time = time.time()
        print(f"{len(dataset)} analysed in {end_time - start_time} seconds")
        result_set = ResultSet(
            model=environment.model,
            ground_truth_dataset=dataset,
            prediction_dataset=result_dataset
        )
        performance = task.compute_performance(result_set)
        return performance

    @flaky(max_runs=1, rerun_filter=rerun_on_flaky_assert())
    def test_training_and_analyse(self):
        """
        Tests for training, analysis, evaluation, model optimization for the task

        Flow of the test:
        - Creates a randomly annotated project with 100 images and a video with 25 frames. Shapes in the project are
            ['rectangle', 'triangle', 'circle']. Annotations are converted to bounding box annotations.
        - Trains a model for 5 epochs. Asserts that the returned model is not a NullModel, that
            validation F-measure is larger than 0.5 and also that OpenVINO optimization runs successfully.
        - Reloads the model in the task and recompute the performance. Asserts that the performance
            difference between the original and the reloaded model is smaller than 1e-4. Ideally there should be no
            difference at all.
        """
        configurable_parameters = self.setup_configurable_parameters()
        _, detection_environment, dataset = self.init_environment(configurable_parameters, 250)
        task = MMObjectDetectionTask(task_environment=detection_environment)
        self.addCleanup(task._delete_scratch_space)

        print("MMDetection task initialized, model training starts.")
        # Train the task.
        # train_task checks that the returned model is not a NullModel, that the task returns an OptimizedModel and that
        # validation f-measure is higher than 0.5, which is a pretty low bar considering that the dataset is so easy

        model = task.train(dataset=dataset)
        self.assertFalse(isinstance(model, NullModel))

        if isinstance(task, IModelOptimizer):
            optimized_models = task.optimize_loaded_model()
            self.assertGreater(len(optimized_models), 0, "Task must return an Optimised model.")
            for m in optimized_models:
                self.assertIsInstance(m, OptimizedModel,
                                      "Optimised model must be an Openvino or DeployableTensorRT model.")

        # Run inference
        validation_performance = self.eval(task, detection_environment, dataset)
        print(f"Evaluated model to have a performance of {validation_performance}")
        self.assertGreater(validation_performance.score.value, 0.5, "Expected F-measure to be higher than 0.5 [flaky]")

        print("Reloading model.")
        # Re-load the model
        task.load_model(task.task_environment)

        print("Reevaluating model.")
        # Performance should be the same after reloading
        performance_after_reloading = self.eval(task, detection_environment, dataset)
        performance_delta = performance_after_reloading.score.value - validation_performance.score.value
        perf_delta_tolerance = 0.0001

        self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f"Expected no or very small performance difference after reloading. Performance delta "
                            f"({validation_performance.score.value} vs {performance_after_reloading.score.value}) was "
                            f"larger than the tolerance of {perf_delta_tolerance}")

        print(f"Performance: {validation_performance.score.value:.4f}")
        print(f"Performance after reloading: {performance_after_reloading.score.value:.4f}")
        print(f"Performance delta after reloading: {performance_delta:.6f}")
