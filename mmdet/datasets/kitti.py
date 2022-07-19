from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import tqdm
from .builder import DATASETS
from .custom import CustomDataset
from PIL import Image


@DATASETS.register_module()
class KittiDataset(CustomDataset):
    def __init__(
        self,
        data_root,
        pipeline,
        classes=None,
        ann_file="label_2",
        img_prefix="image_2",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        file_client_args=dict(backend="disk"),
    ):
        super().__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            proposal_file,
            test_mode,
            filter_empty_gt,
            file_client_args,
        )

    def parse_kitty_coords(
        self,
        file: Path,
        classes: Set[str],
    ) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        result = []
        with open(file, "r") as f:
            for line in f:
                entries = line.split(" ")
                if (
                    len(entries) < 15
                ):  # inferred kitti annotations with tao have 16th datapoint
                    raise Exception(f"Invalid kitty format in {file=} with {entries}")
                try:
                    coords = (
                        entries[0],
                        tuple(map(lambda x: int(float(x)), entries[4 : 4 + 4])),
                        entries[15] if len(entries) > 15 else 1,
                    )
                    # ignore classes we don't care about
                    if coords[0] in classes:
                        result.append(coords)
                except Exception as e:
                    raise Exception(
                        f"Invalid kitty format, cannot parse coords, {file=} with {entries[4:4+4]} and {entries}\n{e}"
                    ) from e
        return result

    def load_annotations(self, label_dir):
        label_dir = Path(label_dir)
        image_dir = Path(self.img_prefix)
        labels = label_dir.glob("*.txt")
        classes_set = self.CLASSES

        dataset = []
        for item in tqdm.tqdm(
            iterable=labels,
            desc=f"Loading dataset: '{self.data_root}'",
        ):
            # maybe to Path("image_2") here
            # image = image_dir / item.relative_to(label_dir).with_suffix(".jpg")
            image: Path = image_dir / item.relative_to(label_dir).with_suffix(".jpg")

            if not image.is_file():
                continue  # skip unknown image file
            with Image.open(image) as img:
                h, w = img.height, img.width

            labels, bboxes, scores = zip(
                *self.parse_kitty_coords(item, classes=classes_set)
            )

            # translate class names to class indexes
            label_dict = {c: i for i, c in enumerate(self.CLASSES)}
            labels = [label_dict[c] for c in labels]

            data = dict(
                filename=image.name,
                height=h,
                width=w,
                ann=dict(
                    bboxes=np.array(bboxes).astype(np.float32),
                    labels=np.array(labels).astype(np.int64),
                    scores=np.array(scores).astype(np.float32),
                ),
            )
            dataset.append(data)

        return dataset

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]

    def get_ann_as_prediction(self):
        """Load inferred tao kitti labels as prediction for mAP computation."""
        result = []
        for ann in self.data_infos:
            ann = ann["ann"]

            pred = [[] for _ in range(len(self.CLASSES))]
            for label, bbox, score in zip(ann["labels"], ann["bboxes"], ann["scores"]):
                pred[label].append([*bbox, score])  # extend with prediction score

            pred = [
                np.empty((0, 5), dtype=np.int32)  # otherwise mAP computation breaks
                if len(classes) == 0
                else np.array(classes).astype(np.float32)
                for classes in pred
            ]
            result.append(pred)

        return result
