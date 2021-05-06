import time
import unittest
import copy
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from flaky import flaky

from noussdk.entities.media_identifier import ImageIdentifier, VideoFrameIdentifier
from noussdk.entities.project import Project
from noussdk.entities.shapes.box import Box
from noussdk.entities.shapes.circle import Circle
from noussdk.entities.shapes.polygon import Polygon
from noussdk.entities.task_environment import TaskEnvironment
from noussdk.usecases.repos import ConfigurableParametersRepo, AnnotationRepo, ImageRepo, VideoRepo
from noussdk.tests.test_helpers import generate_random_annotated_project, generate_and_save_random_annotated_video, \
    generate_training_dataset_of_all_annotated_media_in_project, rerun_on_flaky_assert

from nousapi.apis.detection import MMObjectDetectionTask, MMDetectionParameters
from nousapi.tests.test_helpers import train_task, compute_validation_performance



def convert_annotation_shapes_in_project_to_bounding_boxes(project: Project = None):
    """
    Converts the shapes for all annotations in a project to bounding boxes, so that they can be used to train a
    detection project.

    :param project: Project to convert shapes for

    """
    anno_repo = AnnotationRepo(project)
    image_repo = ImageRepo(project)
    video_repo = VideoRepo(project)

    image_annotation_count = 0
    video_annotation_count = 0

    # Loop over annotations in the project and convert their shapes to bounding boxes
    for annotation in anno_repo.get_all_annotations():
        shapes = copy.deepcopy(annotation.shapes)
        annotation.shapes.clear()
        # Get media belonging to the annotation, in order to be able to extract media dimensions for circle conversion
        media_identifier = annotation.media_identifier

        if isinstance(media_identifier, ImageIdentifier):
            media = image_repo.get_by_id(media_identifier.media_id)
            image_annotation_count += 1
        elif isinstance(media_identifier, VideoFrameIdentifier):
            media = video_repo.get_by_id(media_identifier.media_id)
            video_annotation_count += 1
        else:
            raise NotImplementedError(f"Annotation conversion for media type with identifier {media_identifier} is not "
                                      f"implemented yet.")

        for shape_index, shape in enumerate(shapes):
            # Convert all shapes to bounding boxes
            labels = shape.get_labels(include_empty=True)
            if isinstance(shape, Box):
                continue
            elif isinstance(shape, Circle):
                r, x, y = shape.r, shape.x, shape.y
                aspect_ratio = media.width / media.height
                height_r = aspect_ratio * r
                shapes[shape_index] = Box(x1=x-r, y1=y-height_r, x2=x+r, y2=y+height_r, labels=labels)
            elif isinstance(shape, Polygon):
                shapes[shape_index] = Box(x1=shape.min_x, y1=shape.min_y, x2=shape.max_x, y2=shape.max_y,
                                          labels=labels)

        # Update annotation and persist in repo
        annotation.append_shapes(shapes)
        anno_repo.save(annotation)

    print(f"Converted shapes for {image_annotation_count} image annotations and {video_annotation_count} video "
          f"annotations to bounding boxes.")


class TestMMDetection(unittest.TestCase):
    """
    Collection of MM Detection tests for the MMObjectDetectionTask for running in CI pipeline
    """

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
        project_name = "test_mmdetection_1"
        # Initialize and populate project
        detection_project = generate_random_annotated_project(test_case=self, name=project_name,
                                                              description="test cancel training detection",
                                                              task_name="MMDetection", number_of_images=20,
                                                              image_width=480, image_height=360, max_shapes=100)
        generate_and_save_random_annotated_video(project=detection_project,
                                                 video_name="Video for Detection", width=480, height=360)

        # Convert annotations to detection format
        convert_annotation_shapes_in_project_to_bounding_boxes(detection_project)
        print(f"Project {project_name} created and populated.")

        # Create training dataset and initialize task
        training_dataset = generate_training_dataset_of_all_annotated_media_in_project(detection_project)
        detection_task_node = detection_project.tasks[1]
        detection_environment = TaskEnvironment(project=detection_project,
                                                task_node=detection_task_node,
                                                hardware_resource_configuration=None)
        detection_task = MMObjectDetectionTask(task_environment=detection_environment)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="train_thread")

        # Test stopping after some time
        start_time = time.time()
        train_future = executor.submit(detection_task.train, training_dataset)
        time.sleep(10)  # give train_thread some time to initialize the model
        detection_task.cancel_training()

        # stopping process has to happen in less than 35 seconds
        self.assertLess(time.time() - start_time, 35, "Expected to stop within 35 seconds [flaky].")
        train_future.result()

        # Test stopping immediately
        start_time = time.time()
        train_future = executor.submit(detection_task.train, training_dataset)
        time.sleep(1.0)
        detection_task.cancel_training()

        self.assertLess(time.time() - start_time, 25)  # stopping process has to happen in less than 25 seconds
        train_future.result()

    @flaky(max_runs=3, rerun_filter=rerun_on_flaky_assert())
    def test_training_and_analyse(self):
        """
        Tests for training, analysis, evaluation, model optimization and architecture change for the object detection
        task based on mmdetection

        Flow of the test:
        - Creates a randomly annotated project with 100 images and a video with 25 frames. Shapes in the project are
            ['rectangle', 'triangle', 'circle']. Annotations are converted to bounding box annotations.
        - Trains a faster-rcnn model for 5 epochs. Asserts that the returned model is not a NullModel, that
            validation F-measure is larger than 0.5 and also that OpenVINO optimization runs successfully.
        - Changes the model architecture to cascade-rcnn in the configurable parameters. Updates configurable parameters
            for the task and retrains for 8 epochs. Asserts that the returned model is not a NullModel, that validation
            F-measure is larger than 0.5 and that OpenVINO optimization of cascade-rcnn model runs successfully
        - Reloads the cascade-rcnn model in the task and recompute the performance. Asserts that the performance
            difference between the original and the reloaded model is smaller than 1e-4. Ideally there should be no
            difference at all.
        """
        project_name = "test_mmdetection_2"
        configurable_parameters = MMDetectionParameters()
        configurable_parameters.learning_architecture.model_architecture.value = "faster-rcnn"
        configurable_parameters.learning_parameters.num_epochs.value = 5

        # Initialize and populate project
        detection_project = generate_random_annotated_project(self, name=project_name,
                                                              description="test training and analyse",
                                                              task_name="MMDetection", max_shapes=100,
                                                              number_of_images=100, image_width=1080, image_height=720,
                                                              configurable_parameters=configurable_parameters)
        generate_and_save_random_annotated_video(project=detection_project, number_of_frames=25,
                                                 video_name="Video for Detection tests", width=480, height=360)
        # Convert annotations to detection format
        convert_annotation_shapes_in_project_to_bounding_boxes(detection_project)
        print(f"Project {project_name} created and populated.")

        # Initialize task
        detection_task_node = detection_project.tasks[-1]
        detection_environment = TaskEnvironment(project=detection_project, task_node=detection_task_node)
        task = MMObjectDetectionTask(task_environment=detection_environment)
        self.addCleanup(lambda: task.delete_scratch_space())

        print("MMDetection task initialized, training of Faster-RCNN model starts.")
        # Train the task.
        # train_task checks that the returned model is not a NullModel, that the task returns an OptimizedModel and that
        # validation f-measure is higher than 0.5, which is a pretty low bar considering that the dataset is so easy
        validation_performance_fasterrcnn = train_task(self, task, detection_project, add_video_to_project=False)

        # Change model architecture to cascade-rcnn using the task update interface for configurable parameters.
        # Goal is to verify that the task update interface is working.
        confpar_repo = ConfigurableParametersRepo(detection_project)
        new_configurable_parameters = MMDetectionParameters(confpar_repo.get_latest_by_task_id(detection_task_node.id))
        new_configurable_parameters.learning_architecture.model_architecture.value = "cascade-rcnn"
        new_configurable_parameters.learning_parameters.num_epochs.value = 8
        # Setting results based thresholds to False to for reproducibility. Otherwise reloading and re-evaluating the
        # model for the task can lead to very small performance differences
        new_configurable_parameters.postprocessing.result_based_confidence_threshold.value = False
        new_configurable_parameters.postprocessing.result_based_nms_threshold.value = False
        new_configurable_parameters.postprocessing.confidence_threshold.value = 0.5
        new_configurable_parameters.postprocessing.nms_threshold.value = 0.5
        # Task node has to be set in order to save configurable parameters to the repo
        new_configurable_parameters.task_node = detection_task_node
        confpar_repo.save(new_configurable_parameters)

        # Update configurable parameters using the task update interface
        new_detection_environment = copy.deepcopy(detection_environment)
        new_detection_environment.set_configurable_parameters(new_configurable_parameters)
        task.update_configurable_parameters(new_detection_environment)

        print("Configurable parameters updated. Training of Cascade-RCNN model starts.")
        # Retrain task for cascadercnn architecture.
        validation_performance_cascadercnn = train_task(self, task, detection_project, add_video_to_project=False)

        print("Reloading Cascade-RCNN model.")
        # re-load the model
        task.load_model(task.task_environment)

        # Performance should be the same after reloading
        performance_after_reloading = compute_validation_performance(task, task.task_environment)
        performance_delta = performance_after_reloading.score.value - validation_performance_cascadercnn.score.value
        perf_delta_tolerance = 0.0001

        self.assertLess(np.abs(performance_delta), perf_delta_tolerance,
                        msg=f"Expected no or very small performance difference after reloading. Performance delta was "
                            f"larger than the tolerance of {perf_delta_tolerance}")

        print(f"Faster-rcnn performance: {validation_performance_fasterrcnn.score.value:.4f}")
        print(f"Cascade-rcnn performance: {validation_performance_cascadercnn.score.value:.4f}")
        print(f"Cascade-rcnn performance after reloading: {performance_after_reloading.score.value:.4f}")
        print(f"Performance delta after reloading: {performance_delta:.6f}")
