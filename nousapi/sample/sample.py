import os
import os.path as osp
import sys
import random
from typing import Tuple

import cv2 as cv
from tqdm import tqdm

from mmdet.datasets import CocoDataset
from noussdk.entities.analyse_parameters import AnalyseParameters
from noussdk.entities.annotation import Annotation, AnnotationKind
from noussdk.entities.datasets import Dataset, Subset
from noussdk.entities.image import Image
from noussdk.entities.label import ScoredLabel
from noussdk.entities.project import Project
from noussdk.entities.resultset import ResultSet
from noussdk.entities.shapes.box import Box
from noussdk.entities.task_environment import TaskEnvironment
from noussdk.entities.url import URL
from noussdk.tests.test_helpers import generate_training_dataset_of_all_annotated_media_in_project
from noussdk.usecases.repos import *
from noussdk.utils.project_factory import ProjectFactory

from nousapi.apis.detection import MMObjectDetectionTask


def load_annotation(data_dir, filter_classes=None, dataset_id=0):
    ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.gif')
    def is_valid(filename):
        return not filename.startswith('.') and filename.lower().endswith(ALLOWED_EXTS)

    def find_classes(dir, filter_names=None):
        if filter_names:
            classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in filter_names]
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return class_to_idx

    class_to_idx = find_classes(data_dir, filter_classes)

    out_data = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = osp.join(data_dir, target_class)
        if not osp.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = osp.join(root, fname)
                if is_valid(path):
                    out_data.append((path, class_index, 0, dataset_id, '', -1, -1))\

    if not len(out_data):
        print('Failed to locate images in folder ' + data_dir + f' with extensions {ALLOWED_EXTS}')

    return out_data, class_to_idx


def create_project(projectname, taskname, classes):
    project = ProjectFactory().create_project_single_task(name=projectname, description="",
        label_names=classes, task_name=taskname)
    ProjectRepo().save(project)
    return project


def get_label(x, all_labels):
    label_name = CocoDataset.CLASSES[x]
    return [label for label in all_labels if label.name == label_name][0]

def create_coco_dataset(project):
    pipeline = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]
    coco_dataset = CocoDataset(ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/', pipeline=pipeline)

    print(len(coco_dataset))
    for datum in tqdm(coco_dataset):
        imdata = datum['img']
        imshape = imdata.shape
        image = Image(name=datum['ori_filename'], project=project, numpy=imdata)
        ImageRepo(project).save(image)

        gt_bboxes = datum['gt_bboxes']
        gt_labels = datum['gt_labels']

        shapes = []
        for label, bbox in zip(gt_labels, gt_bboxes):
            project_label = get_label(label, project.get_labels())
            shapes.append(
                Box(x1=float(bbox[0] / imshape[1]),
                    y1=float(bbox[1] / imshape[0]),
                    x2=float(bbox[2] / imshape[1]),
                    y2=float(bbox[3] / imshape[0]),
                    labels=[ScoredLabel(project_label)]))
        annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image.media_identifier, shapes=shapes)
        AnnotationRepo(project).save(annotation)

    # for i, item in enumerate(anno):
    # 	imdata = cv.imread(item[0])
    # 	imdata = cv.cvtColor(imdata, cv.COLOR_RGB2BGR)
    # 	image = Image(name=f"{osp.basename(item[0])}", project=project, numpy=imdata)
    # 	ImageRepo(project).save(image)
    # 	label = [label for label in project.get_labels() if label.name==item[1]][0]
    # 	shapes = [Box.generate_full_box(labels=[ScoredLabel(label)])]
    # 	annotation = Annotation(kind=AnnotationKind.ANNOTATION, media_identifier=image.media_identifier, shapes=shapes)
    # 	AnnotationRepo(project).save(annotation)
    # 	if i > 100**10:
    # 		break

    dataset = generate_training_dataset_of_all_annotated_media_in_project(project)
    DatasetRepo(project).save(dataset)
    print('Dataset generated')
    return dataset


projectname = "MMObjectDetectionSample"
project = create_project(projectname, "MMObjectDetectionTask", CocoDataset.CLASSES)
dataset = create_coco_dataset(project)
print('Tasks:', [task.task_name for task in project.tasks])
print(f"train dataset: {len(dataset.get_subset(Subset.TRAINING))} items")
print(f"validation dataset: {len(dataset.get_subset(Subset.VALIDATION))} items")
environment = TaskEnvironment(project=project, task_node=project.tasks[-1])

task = MMObjectDetectionTask(task_environment=environment)

model = task.train(dataset=dataset)
ModelRepo(project).save(model)
# DatasetRepo(project).save(dataset)


validation_dataset = dataset.get_subset(Subset.VALIDATION)
print(f"validation dataset: {len(validation_dataset)} items")

predicted_validation_dataset = task.analyse(
    validation_dataset.with_empty_annotations(), AnalyseParameters(is_evaluation=True))

resultset = ResultSet(
    model=model,
    ground_truth_dataset=validation_dataset,
    prediction_dataset=predicted_validation_dataset,
)
ResultSetRepo(project).save(resultset)

performance = task.compute_performance(resultset)
resultset.performance = performance
ResultSetRepo(project).save(resultset)

print(resultset.performance)
