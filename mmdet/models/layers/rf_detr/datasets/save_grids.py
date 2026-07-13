# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T  # noqa: N812
from matplotlib.axes import Axes
from supervision import BoxAnnotator, Color, Detections, LabelAnnotator
from torch import Tensor
from torch.utils.data import DataLoader

from mmdet.models.layers.rf_detr.util.box_ops import box_cxcywh_to_xyxy
from mmdet.models.layers.rf_detr.util.logger import get_logger

logger = get_logger()


class DatasetGridSaver:
    """Utility for saving 3x3 image grids to visualize augmentation effects.

    Args:
        data_loader: Dataloader of the dataset to sample images from.
        output_dir: Directory where grid images will be saved.
        max_batches: Number of batches to draw samples from.
        dataset_type: Dataset split label, e.g. ``'train'`` or ``'val'``.
    """

    def __init__(
        self, data_loader: DataLoader, output_dir: Path, max_batches: int = 3, dataset_type: str = "train"
    ) -> None:
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.max_batches = max_batches
        self.dataset_type = dataset_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_grid(self) -> None:
        """Create and save image grids to ``output_dir``.

        Each grid is a 3x3 JPEG containing up to 9 images from a single batch, with bounding boxes and class labels
        drawn on top.
        """
        inv_normalize = T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        box_annotator = BoxAnnotator(thickness=2)
        label_annotator = LabelAnnotator(
            text_color=Color.BLACK,
            text_scale=0.5,
            text_padding=3,
        )

        for batch_idx, (sample, target) in enumerate(self.data_loader):
            if batch_idx >= self.max_batches:
                break

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f"{self.dataset_type} dataset, batch {batch_idx}")
            axes = axes.flatten()

            sample_index = 0
            for sample_index, (single_image, single_target) in enumerate(zip(sample.tensors, target)):
                if sample_index >= 9:
                    break
                self._annotate_and_plot(
                    single_image, single_target, axes[sample_index], inv_normalize, box_annotator, label_annotator
                )

            for i in range(sample_index, 9):
                axes[i].axis("off")

            fig.tight_layout()
            plt.savefig(self.output_dir / f"{self.dataset_type}_batch{batch_idx}_grid.jpg", dpi=200)
            plt.close()

        logger.info(f"Saved {self.dataset_type} grids with augmented images to: {self.output_dir.resolve()}")

    @staticmethod
    def _annotate_and_plot(
        single_image: Tensor,
        single_target: dict[str, Any],
        ax: Axes,
        inv_normalize: T.Normalize,
        box_annotator: BoxAnnotator,
        label_annotator: LabelAnnotator,
    ) -> None:
        """De-normalize a single image tensor, annotate it with boxes and labels, and plot it on ``ax``.

        Args:
            single_image: Normalized image tensor of shape ``(C, H, W)``.
            single_target: Target dict with keys ``'size'``, ``'boxes'`` (cx,cy,wh normalized), and ``'labels'``.
            ax: Matplotlib axis to plot the annotated image on.
            inv_normalize: Inverse normalization transform to convert the tensor back to pixel values.
            box_annotator: ``BoxAnnotator`` instance for drawing bounding boxes.
            label_annotator: ``LabelAnnotator`` instance for drawing class labels.
        """
        from PIL import Image as PILImage

        resized_size = single_target["size"]
        if isinstance(resized_size, Tensor):
            resized_size = resized_size.detach().cpu()
        h, w = int(resized_size[0]), int(resized_size[1])

        de_normalized_img = inv_normalize(single_image)
        if isinstance(de_normalized_img, Tensor):
            de_normalized_img = de_normalized_img.detach().cpu().numpy()
        scene = PILImage.fromarray((np.clip(de_normalized_img.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8))

        if len(single_target["boxes"]) > 0:
            labels_tensor = single_target["labels"]
            if isinstance(labels_tensor, Tensor):
                class_ids = labels_tensor.detach().cpu().numpy().astype(int)
            else:
                class_ids = np.asarray(labels_tensor, dtype=int)

            boxes = single_target["boxes"]
            if isinstance(boxes, Tensor):
                boxes_iter = boxes.detach().cpu()
            else:
                boxes_iter = boxes

            xyxy = np.asarray(
                [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for box in boxes_iter for b in [box_cxcywh_to_xyxy(box)]],
                dtype=np.float32,
            )
            detections = Detections(xyxy=xyxy, class_id=class_ids)
            labels = [str(c) for c in class_ids]
            scene = box_annotator.annotate(scene=scene, detections=detections)
            scene = label_annotator.annotate(scene=scene, detections=detections, labels=labels)

        ax.imshow(scene)
        ax.axis("off")
