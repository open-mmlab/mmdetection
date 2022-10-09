# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import numpy as np
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks


@VISUALIZERS.register_module()
class DetWandbVisualizer(Visualizer):
    """MMDetection Wandb Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        wandb_backend_args (dict, optional): WandB visual backend args.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 wandb_backend_args: Optional[dict] = None,
                 save_dir: Optional[str] = None) -> None:

        if wandb_backend_args is None:
            wandb_backend_args = {}
        vis_backends = [dict(type='WandbVisBackend', **wandb_backend_args)]
        super().__init__(name, image, vis_backends, save_dir)

        # If save_dir is None, the wandb backend will not initialize.
        self.save_dir = save_dir

        self._wandb = None
        init_wandb = True if save_dir is not None else False
        if init_wandb:
            self._wandb = self.get_backend('WandbVisBackend').experiment

        # Set default value. When calling
        # `DetWandbVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}
        self.class_id_to_label = None
        self.class_set = None

        self._record_tables = {}

    def _get_wandb_bboxes(self,
                          bboxes: np.ndarray,
                          labels: np.ndarray,
                          scores: Optional[np.ndarray],
                          log_gt: bool = True) -> dict:
        """Get structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (np.array): Bounding box coordinates in
                (minX, minY, maxX, maxY) format.
            labels (np.array): Label indexes.
            scores (np.array, optional): Detected bbox scores.
            log_gt (bool): Whether to log ground truth or prediction boxes.
                Defaults to True.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        assert len(bboxes) == len(labels)
        if scores is None:
            scores = [None] * len(bboxes)
        else:
            assert len(scores) == len(bboxes)
        wandb_boxes = {}

        box_data = []
        for bbox, label, score in zip(bboxes, labels, scores):
            if not isinstance(label, int):
                label = int(label)
            label = label + 1

            if score is not None:
                score = round(float(score) * 100, 1)
                class_name = self.class_id_to_label[label]
                box_caption = f'{class_name}: {score}'
            else:
                box_caption = str(self.class_id_to_label[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            box_data.append({
                'position': position,
                'class_id': label,
                'box_caption': box_caption,
                'domain': 'pixel'
            })

        wandb_bbox_dict = {
            'box_data': box_data,
            'class_labels': self.class_id_to_label
        }

        if log_gt:
            wandb_boxes['ground_truth'] = wandb_bbox_dict
        else:
            wandb_boxes['predictions'] = wandb_bbox_dict

        return wandb_boxes

    def _get_wandb_masks(self, masks: np.ndarray, labels: np.ndarray) -> dict:
        """Get structured dict for logging masks to W&B.

        Args:
            masks (np.array): Mask datas.
            labels (np.array): Label ids.

        Returns:
            Dictionary of masks to be logged.
        """
        mask_label_dict = dict()
        for mask, label in zip(masks, labels):
            label = label + 1
            # Create composite masks for each class.
            if label not in mask_label_dict.keys():
                mask_label_dict[label] = mask
            else:
                mask_label_dict[label] = np.logical_or(mask_label_dict[label],
                                                       mask)

        wandb_masks = dict()
        for key, value in mask_label_dict.items():
            # Create mask for that class.
            value = value.astype(np.uint8)
            value[value > 0] = key

            # Create dict of masks for logging.
            class_name = self.class_id_to_label[key]
            wandb_masks[class_name] = {
                'mask_data': value,
                'class_labels': self.class_id_to_label
            }
        return wandb_masks

    def _draw_instances(self,
                        image: np.ndarray,
                        instances: InstanceData,
                        log_gt=True):
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            wandb.Image: the WandB drawn image.
        """
        data_dict = {}
        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels
            scores = instances.get('scores', None)
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels, scores, log_gt=log_gt)
            data_dict['boxes'] = wandb_boxes

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks

            if isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            wandb_masks = self._get_wandb_masks(masks, labels)
            data_dict['masks'] = wandb_masks

        img_data = self._wandb.Image(
            image, classes=self.class_set, **data_dict)

        return img_data

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        Note: Considering the particularity of WandB, the ``show``,
        ``wait_time`` and ``out_file`` are invalid.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image.
                Default to False. Invalid.
            wait_time (float): The interval of show (s). Defaults to 0.
                Invalid.
            out_file (str): Path to output file. Defaults to None.
                Invalid.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        if self.save_dir is None:
            warnings.warn('`WandbVisBackend` is not initialized '
                          'because save_dir is None. Return directly.')
            return
        if show is True:
            warnings.warn('Wandb Visualizer does not have a local window '
                          'display function, ignore the show parameter.')
        if out_file is not None:
            warnings.warn('Wandb Visualizer does not have the function '
                          'of saving files to the local disk, '
                          'ignore the out_file parameter.')

        if self.class_id_to_label is None:
            assert 'CLASSES' in self.dataset_meta
            self.class_id_to_label = {
                id + 1: name
                for id, name in enumerate(self.dataset_meta['CLASSES'])
            }
            self.class_set = self._wandb.Classes([{
                'id': id,
                'name': name
            } for id, name in self.class_id_to_label.items()])

        if data_sample is not None:
            data_sample = data_sample.numpy()

        gt_img_data = None
        pred_img_data = None

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances)
            # TODO: Support panoptic_seg

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(
                    image, pred_instances, log_gt=False)
            # TODO: Support panoptic_seg

        if name in self._record_tables:
            table = self._record_tables[name]
        else:
            if gt_img_data is not None and pred_img_data is not None:
                table = self._wandb.Table(
                    columns=['ground_truth', 'prediction'])
            elif gt_img_data is not None:
                table = self._wandb.Table(columns=['ground_truth'])
            else:
                table = self._wandb.Table(columns=['prediction'])
            self._record_tables[name] = table

        if gt_img_data is not None and pred_img_data is not None:
            table.add_data(gt_img_data, pred_img_data)
        elif gt_img_data is not None:
            table.add_data(gt_img_data)
        else:
            table.add_data(pred_img_data)

    def close(self) -> None:
        """close an opened object."""
        if self._wandb is not None:
            self._log_all_tables()
        super().close()

    def _log_all_tables(self) -> None:
        """Log the W&B Tables."""
        for name, table in self._record_tables.items():
            wandb_artifact = self._wandb.Artifact(name, type='wandb')
            wandb_artifact.add(table, name)
            self._wandb.run.log_artifact(wandb_artifact)
