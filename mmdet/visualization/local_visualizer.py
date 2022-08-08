# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine import Visualizer
from mmengine.data import InstanceData, PixelData
from mmengine.dist import master_only

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import bitmap_to_polygon


def palette_val(palette: List[tuple]) -> List[tuple]:
    """Convert palette to matplotlib palette.

    Args:
        palette (List[tuple]): A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(palette: Union[List[tuple], str, tuple],
                num_classes: int) -> List[Tuple[int]]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple): palette inputs.
        num_classes (int): the number of classes.

    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        from mmdet.datasets import CocoDataset, CocoPanopticDataset
        dataset_palette = CocoDataset.METAINFO['PALETTE']
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.METAINFO['PALETTE']
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        dataset_palette = CityscapesDataset.METAINFO['PALETTE']
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        dataset_palette = VOCDataset.METAINFO['PALETTE']
    elif mmcv.is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette


def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.data import InstanceData
        >>> from mmdet.structures import DetDataSample
        >>> from mmdet.visualization import DetLocalVisualizer

        >>> det_local_visualizer = DetLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_instances.labels = torch.randint(0, 2, (1,))
        >>> gt_det_data_sample = DetDataSample()
        >>> gt_det_data_sample.gt_instances = gt_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample)
        >>> det_local_visualizer.add_datasample(
        ...                       'image', image, gt_det_data_sample,
        ...                        out_file='out_file.jpg')
        >>> det_local_visualizer.add_datasample(
        ...                        'image', image, gt_det_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_det_data_sample = DetDataSample()
        >>> pred_det_data_sample.pred_instances = pred_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample,
        ...                         pred_det_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes[:, :2] + self.line_width
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [mask_palette[label] for label in labels]

            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

            if 'bboxes' not in instances or instances.bboxes.sum() == 0:
                # instances.bboxes.sum()==0 represent dummy bboxes.
                # A typical example of SOLO does not exist bbox branch.
                areas = []
                positions = []
                for mask in masks:
                    _, _, stats, centroids = cv2.connectedComponentsWithStats(
                        mask.astype(np.uint8), connectivity=8)
                    largest_id = np.argmax(stats[1:, -1]) + 1
                    positions.append(centroids[largest_id])
                    areas.append(stats[largest_id, -1])
                areas = np.stack(areas, axis=0)
                scales = _get_adaptive_scales(areas)

                for i, (pos, label) in enumerate(zip(positions, labels)):
                    label_text = classes[
                        label] if classes is not None else f'class {label}'
                    if 'scores' in instances:
                        score = round(float(instances.scores[i]) * 100, 1)
                        label_text += f': {score}'

                    self.draw_texts(
                        label_text,
                        pos,
                        colors=text_colors[i],
                        font_sizes=int(13 * scales[i]),
                        horizontal_alignments='center',
                        bboxes=[{
                            'facecolor': 'black',
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])
        return self.get_image()

    def _draw_panoptic_seg(self, image: np.ndarray,
                           panoptic_seg: ['PixelData'],
                           classes: Optional[List[str]]) -> np.ndarray:
        """Draw panoptic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            panoptic_seg (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            classes (List[str], optional): Category information.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        # TODO: Is there a way to bypass？
        num_classes = len(classes)

        panoptic_seg = panoptic_seg.sem_seg[0]
        ids = np.unique(panoptic_seg)[::-1]
        legal_indices = ids != num_classes  # for VOID label
        ids = ids[legal_indices]

        labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
        segms = (panoptic_seg[None] == ids[:, None, None])

        max_label = int(max(labels) if len(labels) > 0 else 0)
        mask_palette = get_palette(self.mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]

        self.set_image(image)

        # draw segm
        polygons = []
        for i, mask in enumerate(segms):
            contours, _ = bitmap_to_polygon(mask)
            polygons.extend(contours)
        self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
        self.draw_binary_masks(segms, colors=colors, alphas=self.alpha)

        # draw label
        areas = []
        positions = []
        for mask in segms:
            _, _, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8)
            max_id = np.argmax(stats[1:, -1]) + 1
            positions.append(centroids[max_id])
            areas.append(stats[max_id, -1])
        areas = np.stack(areas, axis=0)
        scales = _get_adaptive_scales(areas)

        text_palette = get_palette(self.text_color, max_label + 1)
        text_colors = [text_palette[label] for label in labels]

        for i, (pos, label) in enumerate(zip(positions, labels)):
            label_text = classes[label]

            self.draw_texts(
                label_text,
                pos,
                colors=text_colors[i],
                font_sizes=int(13 * scales[i]),
                bboxes=[{
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                }],
                horizontal_alignments='center')
        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            gt_sample: Optional['DetDataSample'] = None,
            pred_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`DetDataSample`, optional): GT DetDataSample.
                Defaults to None.
            pred_sample (:obj:`DetDataSample`, optional): Prediction
                DetDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = self.dataset_meta.get('CLASSES', None)
        palette = self.dataset_meta.get('PALETTE', None)

        gt_img_data = None
        pred_img_data = None

        if draw_gt and gt_sample is not None:
            gt_img_data = image
            if 'gt_instances' in gt_sample:
                gt_img_data = self._draw_instances(image,
                                                   gt_sample.gt_instances,
                                                   classes, palette)

            if 'gt_panoptic_seg' in gt_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, gt_sample.gt_panoptic_seg, classes)

        if draw_pred and pred_sample is not None:
            pred_img_data = image
            if 'pred_instances' in pred_sample:
                pred_instances = pred_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)
            if 'pred_panoptic_seg' in pred_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data,
                    pred_sample.pred_panoptic_seg.cpu().numpy(), classes)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
