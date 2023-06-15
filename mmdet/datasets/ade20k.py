# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmengine import fileio

from mmdet.registry import DATASETS
from .base_semseg_dataset import BaseSegDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset

ADE_PALETTE = [(120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
               (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
               (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
               (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
               (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
               (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
               (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
               (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
               (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
               (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
               (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
               (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
               (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
               (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
               (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
               (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
               (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
               (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
               (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
               (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
               (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
               (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
               (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
               (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
               (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
               (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
               (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
               (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
               (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
               (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
               (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
               (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
               (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
               (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
               (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
               (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
               (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
               (102, 255, 0), (92, 0, 255)]


@DATASETS.register_module()
class ADE20KPanopticDataset(CocoPanopticDataset):
    METAINFO = {
        'classes':
        ('bed', 'window', 'cabinet', 'person', 'door', 'table', 'curtain',
         'chair', 'car', 'painting, picture', 'sofa', 'shelf', 'mirror',
         'armchair', 'seat', 'fence', 'desk', 'wardrobe, closet, press',
         'lamp', 'tub', 'rail', 'cushion', 'box', 'column, pillar',
         'signboard, sign', 'chest of drawers, chest, bureau, dresser',
         'counter', 'sink', 'fireplace', 'refrigerator, icebox', 'stairs',
         'case, display case, showcase, vitrine',
         'pool table, billiard table, snooker table', 'pillow',
         'screen door, screen', 'bookcase', 'coffee table',
         'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower',
         'book', 'bench', 'countertop', 'stove', 'palm, palm tree',
         'kitchen island', 'computer', 'swivel chair', 'boat',
         'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier',
         'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv',
         'airplane', 'clothes', 'pole',
         'bannister, banister, balustrade, balusters, handrail',
         'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'van', 'ship',
         'fountain', 'washer, automatic washer, washing machine',
         'plaything, toy', 'stool', 'barrel, cask', 'basket, handbasket',
         'bag', 'minibike, motorbike', 'oven', 'ball', 'food, solid food',
         'step, stair', 'trade name', 'microwave', 'pot', 'animal', 'bicycle',
         'dishwasher', 'screen', 'sculpture', 'hood, exhaust hood', 'sconce',
         'vase', 'traffic light', 'tray', 'trash can', 'fan', 'plate',
         'monitor', 'bulletin board', 'radiator', 'glass, drinking glass',
         'clock', 'flag', 'wall', 'building', 'sky', 'floor', 'tree',
         'ceiling', 'road, route', 'grass', 'sidewalk, pavement',
         'earth, ground', 'mountain, mount', 'plant', 'water', 'house', 'sea',
         'rug', 'field', 'rock, stone', 'base, pedestal, stand', 'sand',
         'skyscraper', 'grandstand, covered stand', 'path', 'runway',
         'stairway, staircase', 'river', 'bridge, span', 'blind, screen',
         'hill', 'bar', 'hovel, hut, hutch, shack, shanty', 'tower',
         'dirt track', 'land, ground, soil',
         'escalator, moving staircase, moving stairway',
         'buffet, counter, sideboard',
         'poster, posting, placard, notice, bill, card', 'stage',
         'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
         'canopy', 'pool', 'falls', 'tent', 'cradle', 'tank, storage tank',
         'lake', 'blanket, cover', 'pier', 'crt screen', 'shower'),
        'thing_classes':
        ('bed', 'window', 'cabinet', 'person', 'door', 'table', 'curtain',
         'chair', 'car', 'painting, picture', 'sofa', 'shelf', 'mirror',
         'armchair', 'seat', 'fence', 'desk', 'wardrobe, closet, press',
         'lamp', 'tub', 'rail', 'cushion', 'box', 'column, pillar',
         'signboard, sign', 'chest of drawers, chest, bureau, dresser',
         'counter', 'sink', 'fireplace', 'refrigerator, icebox', 'stairs',
         'case, display case, showcase, vitrine',
         'pool table, billiard table, snooker table', 'pillow',
         'screen door, screen', 'bookcase', 'coffee table',
         'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower',
         'book', 'bench', 'countertop', 'stove', 'palm, palm tree',
         'kitchen island', 'computer', 'swivel chair', 'boat',
         'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier',
         'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv',
         'airplane', 'clothes', 'pole',
         'bannister, banister, balustrade, balusters, handrail',
         'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'van', 'ship',
         'fountain', 'washer, automatic washer, washing machine',
         'plaything, toy', 'stool', 'barrel, cask', 'basket, handbasket',
         'bag', 'minibike, motorbike', 'oven', 'ball', 'food, solid food',
         'step, stair', 'trade name', 'microwave', 'pot', 'animal', 'bicycle',
         'dishwasher', 'screen', 'sculpture', 'hood, exhaust hood', 'sconce',
         'vase', 'traffic light', 'tray', 'trash can', 'fan', 'plate',
         'monitor', 'bulletin board', 'radiator', 'glass, drinking glass',
         'clock', 'flag'),
        'stuff_classes':
        ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route',
         'grass', 'sidewalk, pavement', 'earth, ground', 'mountain, mount',
         'plant', 'water', 'house', 'sea', 'rug', 'field', 'rock, stone',
         'base, pedestal, stand', 'sand', 'skyscraper',
         'grandstand, covered stand', 'path', 'runway', 'stairway, staircase',
         'river', 'bridge, span', 'blind, screen', 'hill', 'bar',
         'hovel, hut, hutch, shack, shanty', 'tower', 'dirt track',
         'land, ground, soil', 'escalator, moving staircase, moving stairway',
         'buffet, counter, sideboard',
         'poster, posting, placard, notice, bill, card', 'stage',
         'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
         'canopy', 'pool', 'falls', 'tent', 'cradle', 'tank, storage tank',
         'lake', 'blanket, cover', 'pier', 'crt screen', 'shower'),
        'palette':
        ADE_PALETTE
    }


@DATASETS.register_module()
class ADE20KInstanceDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('bed', 'windowpane', 'cabinet', 'person', 'door', 'table', 'curtain',
         'chair', 'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair',
         'seat', 'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 'railing',
         'cushion', 'box', 'column', 'signboard', 'chest of drawers',
         'counter', 'sink', 'fireplace', 'refrigerator', 'stairs', 'case',
         'pool table', 'pillow', 'screen door', 'bookcase', 'coffee table',
         'toilet', 'flower', 'book', 'bench', 'countertop', 'stove', 'palm',
         'kitchen island', 'computer', 'swivel chair', 'boat',
         'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier',
         'awning', 'streetlight', 'booth', 'television receiver', 'airplane',
         'apparel', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship',
         'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag',
         'minibike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave',
         'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture',
         'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan',
         'plate', 'monitor', 'bulletin board', 'radiator', 'glass', 'clock',
         'flag'),
        'palette': [(204, 5, 255), (230, 230, 230), (224, 5, 255),
                    (150, 5, 61), (8, 255, 51), (255, 6, 82), (255, 51, 7),
                    (204, 70, 3), (0, 102, 200), (255, 6, 51), (11, 102, 255),
                    (255, 7, 71), (220, 220, 220), (8, 255, 214),
                    (7, 255, 224), (255, 184, 6), (10, 255, 71), (7, 255, 255),
                    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                    (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255),
                    (235, 12, 255), (0, 163, 255), (250, 10, 15), (20, 255, 0),
                    (255, 224, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255),
                    (0, 173, 255), (0, 255, 245), (0, 255, 112), (0, 255, 133),
                    (255, 0, 0), (255, 163, 0), (194, 255, 0), (0, 143, 255),
                    (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173),
                    (10, 0, 255), (173, 255, 0), (255, 92, 0), (255, 0, 245),
                    (255, 0, 102), (255, 173, 0), (255, 0, 20), (0, 31, 255),
                    (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194),
                    (0, 255, 82), (0, 112, 255), (51, 0, 255), (0, 122, 255),
                    (255, 153, 0), (0, 255, 10), (163, 255, 0), (255, 235, 0),
                    (8, 184, 170), (184, 0, 255), (255, 0, 31), (0, 214, 255),
                    (255, 0, 112), (92, 255, 0), (70, 184, 160), (163, 0, 255),
                    (71, 255, 0), (255, 0, 163), (255, 204, 0), (255, 0, 143),
                    (133, 255, 0), (255, 0, 235), (245, 0, 255), (255, 0, 122),
                    (255, 245, 0), (214, 255, 0), (0, 204, 255), (255, 255, 0),
                    (0, 153, 255), (0, 41, 255), (0, 255, 204), (41, 0, 255),
                    (41, 255, 0), (173, 0, 255), (0, 245, 255), (0, 255, 184),
                    (0, 92, 255), (184, 255, 0), (255, 214, 0), (25, 194, 194),
                    (102, 255, 0), (92, 0, 255)],
    }


@DATASETS.register_module()
class ADE20KSegDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=ADE_PALETTE)

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 return_classes=False,
                 **kwargs) -> None:
        self.return_classes = return_classes
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            List[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            if self.return_classes:
                data_info['text'] = list(self._metainfo['classes'])
            data_list.append(data_info)
        return data_list
