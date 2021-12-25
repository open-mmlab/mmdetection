import numpy as np


def get_palette(palette, num_classes=None):
    if palette == 'coco':
        from mmdet.datasets import CocoDataset
        palette = CocoDataset.PALETTE
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        palette = VOCDataset.PALETTE
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        palette = CityscapesDataset.PALETTE
    elif palette == 'random' or palette is None:
        assert isinstance(num_classes, int)
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        palette = [tuple(c) for c in palette]
        np.random.set_state(state)
    return palette
