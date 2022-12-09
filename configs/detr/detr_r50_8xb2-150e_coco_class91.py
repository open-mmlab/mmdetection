_base_ = ['./detr_r50_8xb2-150e_coco.py']

model = dict(bbox_head=dict(num_classes=91))

metainfo = dict(
    CLASSES=(None, 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', None,
             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
             'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
             None, 'backpack', 'umbrella', None, None, 'handbag', 'tie',
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
             'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', None, 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
             'chair', 'couch', 'potted plant', 'bed', None, 'dining table',
             None, None, 'toilet', None, 'tv', 'laptop', 'mouse', 'remote',
             'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', None, 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush'),
    PALETTE=[
        None, (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
        (250, 170, 30), (100, 170, 30), None, (220, 220, 0), (175, 116, 175),
        (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252),
        (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0),
        (174, 57, 255), (199, 100, 0), (72, 0, 118), None, (255, 179, 240),
        (0, 125, 92), None, None, (209, 0, 151),
        (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73),
        (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243),
        (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186),
        (197, 226, 255), None, (171, 134, 1), (109, 63, 54), (207, 138, 255),
        (151, 0, 95), (9, 80, 61), (84, 105, 51),
        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
        (163, 255, 0), (119, 0, 170), None, (0, 182, 199), None, None,
        (0, 165, 120), None, (183, 130, 88), (95, 32, 0), (130, 114, 135),
        (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114),
        (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106), None,
        (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255),
        (201, 57, 1), (246, 0, 122), (191, 162, 208)
    ]  # Used for visualization.
)

train_dataloader = dict(dataset=dict(metainfo=metainfo))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))
