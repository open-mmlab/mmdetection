# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .coco import CocoDataset

# images exist in annotations but not in image folder.
objv2_ignore_list = [
    osp.join('patch16', 'objects365_v2_00908726.jpg'),
    osp.join('patch6', 'objects365_v1_00320532.jpg'),
    osp.join('patch6', 'objects365_v1_00320534.jpg'),
]


@DATASETS.register_module()
class Objects365V1Dataset(CocoDataset):
    """Objects365 v1 dataset for detection."""

    METAINFO = {
        'classes':
        ('person', 'sneakers', 'chair', 'hat', 'lamp', 'bottle',
         'cabinet/shelf', 'cup', 'car', 'glasses', 'picture/frame', 'desk',
         'handbag', 'street lights', 'book', 'plate', 'helmet',
         'leather shoes', 'pillow', 'glove', 'potted plant', 'bracelet',
         'flower', 'tv', 'storage box', 'vase', 'bench', 'wine glass', 'boots',
         'bowl', 'dining table', 'umbrella', 'boat', 'flag', 'speaker',
         'trash bin/can', 'stool', 'backpack', 'couch', 'belt', 'carpet',
         'basket', 'towel/napkin', 'slippers', 'barrel/bucket', 'coffee table',
         'suv', 'toy', 'tie', 'bed', 'traffic light', 'pen/pencil',
         'microphone', 'sandals', 'canned', 'necklace', 'mirror', 'faucet',
         'bicycle', 'bread', 'high heels', 'ring', 'van', 'watch', 'sink',
         'horse', 'fish', 'apple', 'camera', 'candle', 'teddy bear', 'cake',
         'motorcycle', 'wild bird', 'laptop', 'knife', 'traffic sign',
         'cell phone', 'paddle', 'truck', 'cow', 'power outlet', 'clock',
         'drum', 'fork', 'bus', 'hanger', 'nightstand', 'pot/pan', 'sheep',
         'guitar', 'traffic cone', 'tea pot', 'keyboard', 'tripod', 'hockey',
         'fan', 'dog', 'spoon', 'blackboard/whiteboard', 'balloon',
         'air conditioner', 'cymbal', 'mouse', 'telephone', 'pickup truck',
         'orange', 'banana', 'airplane', 'luggage', 'skis', 'soccer',
         'trolley', 'oven', 'remote', 'baseball glove', 'paper towel',
         'refrigerator', 'train', 'tomato', 'machinery vehicle', 'tent',
         'shampoo/shower gel', 'head phone', 'lantern', 'donut',
         'cleaning products', 'sailboat', 'tangerine', 'pizza', 'kite',
         'computer box', 'elephant', 'toiletries', 'gas stove', 'broccoli',
         'toilet', 'stroller', 'shovel', 'baseball bat', 'microwave',
         'skateboard', 'surfboard', 'surveillance camera', 'gun', 'life saver',
         'cat', 'lemon', 'liquid soap', 'zebra', 'duck', 'sports car',
         'giraffe', 'pumpkin', 'piano', 'stop sign', 'radiator', 'converter',
         'tissue ', 'carrot', 'washing machine', 'vent', 'cookies',
         'cutting/chopping board', 'tennis racket', 'candy',
         'skating and skiing shoes', 'scissors', 'folder', 'baseball',
         'strawberry', 'bow tie', 'pigeon', 'pepper', 'coffee machine',
         'bathtub', 'snowboard', 'suitcase', 'grapes', 'ladder', 'pear',
         'american football', 'basketball', 'potato', 'paint brush', 'printer',
         'billiards', 'fire hydrant', 'goose', 'projector', 'sausage',
         'fire extinguisher', 'extension cord', 'facial mask', 'tennis ball',
         'chopsticks', 'electronic stove and gas stove', 'pie', 'frisbee',
         'kettle', 'hamburger', 'golf club', 'cucumber', 'clutch', 'blender',
         'tong', 'slide', 'hot dog', 'toothbrush', 'facial cleanser', 'mango',
         'deer', 'egg', 'violin', 'marker', 'ship', 'chicken', 'onion',
         'ice cream', 'tape', 'wheelchair', 'plum', 'bar soap', 'scale',
         'watermelon', 'cabbage', 'router/modem', 'golf ball', 'pine apple',
         'crane', 'fire truck', 'peach', 'cello', 'notepaper', 'tricycle',
         'toaster', 'helicopter', 'green beans', 'brush', 'carriage', 'cigar',
         'earphone', 'penguin', 'hurdle', 'swing', 'radio', 'CD',
         'parking meter', 'swan', 'garlic', 'french fries', 'horn', 'avocado',
         'saxophone', 'trumpet', 'sandwich', 'cue', 'kiwi fruit', 'bear',
         'fishing rod', 'cherry', 'tablet', 'green vegetables', 'nuts', 'corn',
         'key', 'screwdriver', 'globe', 'broom', 'pliers', 'volleyball',
         'hammer', 'eggplant', 'trophy', 'dates', 'board eraser', 'rice',
         'tape measure/ruler', 'dumbbell', 'hamimelon', 'stapler', 'camel',
         'lettuce', 'goldfish', 'meat balls', 'medal', 'toothpaste',
         'antelope', 'shrimp', 'rickshaw', 'trombone', 'pomegranate',
         'coconut', 'jellyfish', 'mushroom', 'calculator', 'treadmill',
         'butterfly', 'egg tart', 'cheese', 'pig', 'pomelo', 'race car',
         'rice cooker', 'tuba', 'crosswalk sign', 'papaya', 'hair drier',
         'green onion', 'chips', 'dolphin', 'sushi', 'urinal', 'donkey',
         'electric drill', 'spring rolls', 'tortoise/turtle', 'parrot',
         'flute', 'measuring cup', 'shark', 'steak', 'poker card',
         'binoculars', 'llama', 'radish', 'noodles', 'yak', 'mop', 'crab',
         'microscope', 'barbell', 'bread/bun', 'baozi', 'lion', 'red cabbage',
         'polar bear', 'lighter', 'seal', 'mangosteen', 'comb', 'eraser',
         'pitaya', 'scallop', 'pencil case', 'saw', 'table tennis paddle',
         'okra', 'starfish', 'eagle', 'monkey', 'durian', 'game board',
         'rabbit', 'french horn', 'ambulance', 'asparagus', 'hoverboard',
         'pasta', 'target', 'hotair balloon', 'chainsaw', 'lobster', 'iron',
         'flashlight'),
        'palette':
        None
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        # 'categories' list in objects365_train.json and objects365_val.json
        # is inconsistent, need sort list(or dict) before get cat_ids.
        cats = self.coco.cats
        sorted_cats = {i: cats[i] for i in sorted(cats)}
        self.coco.cats = sorted_cats
        categories = self.coco.dataset['categories']
        sorted_categories = sorted(categories, key=lambda i: i['id'])
        self.coco.dataset['categories'] = sorted_categories
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list


@DATASETS.register_module()
class Objects365V2Dataset(CocoDataset):
    """Objects365 v2 dataset for detection."""
    METAINFO = {
        'classes':
        ('Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp',
         'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
         'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
         'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower',
         'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots',
         'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt',
         'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker',
         'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool',
         'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum',
         'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle',
         'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned',
         'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel',
         'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed',
         'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple',
         'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck',
         'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock',
         'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger',
         'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine',
         'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle',
         'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane',
         'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage',
         'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone',
         'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane',
         'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
         'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza',
         'Elephant', 'Skateboard', 'Surfboard', 'Gun',
         'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot',
         'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper',
         'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
         'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board',
         'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder',
         'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball',
         'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle',
         'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
         'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club',
         'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear',
         'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong',
         'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask',
         'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide',
         'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee',
         'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon',
         'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon',
         'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog',
         'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer',
         'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple',
         'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle',
         'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone',
         'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion',
         'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom',
         'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
         'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese',
         'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue',
         'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap',
         'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut',
         'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
         'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate',
         'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker',
         'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
         'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin',
         'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill',
         'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi',
         'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case',
         'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop',
         'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle',
         'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster',
         'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling',
         'Table Tennis '),
        'palette':
        None
    }

    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            # file_name should be `patchX/xxx.jpg`
            file_name = osp.join(
                osp.split(osp.split(raw_img_info['file_name'])[0])[-1],
                osp.split(raw_img_info['file_name'])[-1])

            if file_name in objv2_ignore_list:
                continue

            raw_img_info['file_name'] = file_name
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
