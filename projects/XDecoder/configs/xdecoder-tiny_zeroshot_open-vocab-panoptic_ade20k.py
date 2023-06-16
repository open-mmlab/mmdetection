_base_ = [
    '_base_/xdecoder-tiny_open-vocab-panoptic.py',
    'mmdet::_base_/datasets/ade20k_panoptic.py'
]

model = dict(test_cfg=dict(mask_thr=0.4))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=_base_.backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'stuff_text'))
]

x_decoder_ade20k_thing_classes = (
    'bed', 'window', 'cabinet', 'person', 'door', 'table', 'curtain', 'chair',
    'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair', 'seat', 'fence',
    'desk', 'wardrobe', 'lamp', 'tub', 'rail', 'cushion', 'box', 'column',
    'signboard', 'chest of drawers', 'counter', 'sink', 'fireplace',
    'refrigerator', 'stairs', 'case', 'pool table', 'pillow', 'screen door',
    'bookcase', 'coffee table', 'toilet', 'flower', 'book', 'bench',
    'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'arcade machine', 'bus', 'towel', 'light', 'truck',
    'chandelier', 'awning', 'street lamp', 'booth', 'tv', 'airplane',
    'clothes', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship',
    'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag',
    'minibike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave',
    'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture', 'hood',
    'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'plate',
    'monitor', 'bulletin board', 'radiator', 'glass', 'clock', 'flag')

x_decoder_ade20k_stuff_classes = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'grass',
    'sidewalk', 'earth', 'mountain', 'plant', 'water', 'house', 'sea', 'rug',
    'field', 'rock', 'base', 'sand', 'skyscraper', 'grandstand', 'path',
    'runway', 'stairway', 'river', 'bridge', 'blind', 'hill', 'bar', 'hovel',
    'tower', 'dirt track', 'land', 'escalator', 'buffet', 'poster', 'stage',
    'conveyer belt', 'canopy', 'pool', 'falls', 'tent', 'cradle', 'tank',
    'lake', 'blanket', 'pier', 'crt screen', 'shower')

val_dataloader = dict(
    dataset=dict(
        metainfo=dict(
            thing_classes=x_decoder_ade20k_thing_classes,
            stuff_classes=x_decoder_ade20k_stuff_classes),
        return_classes=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
