"""
Dataset class for parsing aces-vision-like annotated dataset.
"""
import json
from pathlib import Path

import mmcv
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose


@DATASETS.register_module()
class AcesVisionDataset(CustomDataset):

    def __init__(self,
                 pipeline,
                 dataset_path=None,
                 split='train',
                 test_mode=False,
                 filter_empty_gt=True,
                 ):
        """
        Args:
            pipeline (callable): A preprocess pipeline defined
                    in the configuration file.
            dataset_path (str or pathlib.Path): A path to a dataset.
            split (str): A name of a split file, e.g., train, val, test.
                default = 'train'
            test_mode (bool): If set True, annotation will not be loaded
                    (load only images).
                default = False
            filter_empty_gt (bool): If set True and `test_mode=False`,
                    images without bounding boxes will be filtered out.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.file_client = mmcv.FileClient()  # default backend
        self.CLASSES = self.get_classes()

        # for aces-vision dataset, the following variables will not be used
        self.img_prefix = self.seg_prefix = None
        self.proposal_file = self.proposals = None

        # load annotations (and proposals)
        self.data_infos = self.load_annotations()

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def get_classes(self):
        """Get class names from label.txt.

        Returns:
            classes (List[str]): Names of classes.
        """
        cls_path = Path(self.dataset_path) / 'cfg' / 'label.txt'
        assert cls_path, 'label.txt does not exist.'

        with open(cls_path) as f:
            classes = [line.strip() for line in f.readlines()]

        return classes

    def load_annotations(self):
        """Load annotations.

        Returns:
            data_infos (List[dict]): Annotations for each image.
                    The inside dict have the following keys:
                    filename (pathlib.Path): Path to image.
                    width (int): Image width.
                    height (int): Image height.
                    ann (Dict[numpy.ndarray]): Annotations in image.
                            It has the following keys:
                        bboxes (numpy.ndarray): BBox coordinates like
                                (x1, y1, x2, y2).
                            dtype = numpy.float32
                            shape = (num_obj, 4)
                        labels (numpy.ndarray): Labels for bboxes.
                            dtype = numpy.int64
                            shape = (num_obj, )
        """
        data_list = Path(self.dataset_path) / 'cfg' / f'{self.split}.txt'
        assert data_list.exists(), f'{self.split}.txt does not exist.'

        with open(data_list) as f:
            filenames = [line.strip() for line in f.readlines()]

        data_infos = []
        for filename in tqdm(filenames, desc='Load annotations'):
            file_path = self.dataset_path / 'data' / filename

            anno_path = file_path.with_suffix('.json')
            with open(anno_path, 'r') as f:
                annos = json.load(f)

            if len(annos) == 0:
                continue

            bboxes, labels = [], []
            for anno in annos:
                x1, y1, w, h = anno['bbox']
                x2, y2 = x1 + w, y1 + h
                bbox = [x1, y1, x2, y2]
                bboxes.append(bbox)
                labels.append(anno['category_id'])
            ann = dict(
                bboxes=np.array(bboxes, dtype=np.float32),
                labels=np.array(labels, dtype=np.int64),
            )

            # get image size
            img_path = file_path.with_suffix('.jpg')
            height, width = mmcv.imread(img_path).shape[:2]

            data_info = dict(
                filename=img_path,
                width=width,
                height=height,
                ann=ann,
            )
            data_infos.append(data_info)

        return data_infos
