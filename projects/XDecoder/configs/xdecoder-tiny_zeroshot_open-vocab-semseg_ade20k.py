_base_ = ['mmdet::_base_/default_runtime.py', '_base_/xdecoder-tiny.py']

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=640,
        keep_ratio=True,
        short_side_mode=True,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='LoadSemSegAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'seg_map_path', 'img',
                   'gt_seg_map', 'text'))
]

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'

x_decoder_ade20k_classes = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'window', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'tub',
    'rail', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning', 'street lamp', 'booth',
    'tv', 'airplane', 'dirt track', 'clothes', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
    'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'pool', 'stool', 'barrel', 'basket', 'falls', 'tent', 'bag', 'minibike',
    'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
    'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
    'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray',
    'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
    'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=x_decoder_ade20k_classes),
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        return_classes=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='SemSegMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
