import numpy as np

import mmdet.datasets as datasets


def get_palette(palette, num_classes=None):
    if palette == 'coco':
        palette = datasets.CocoDataset.PALETTE
    elif palette == 'voc':
        palette = datasets.VOCDataset.PALETTE
    elif palette == 'citys':
        palette = datasets.CityscapesDataset.PALETTE
    elif palette == 'random' or palette is None:
        assert isinstance(num_classes, int)
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        palette = [tuple(c) for c in palette]
        np.random.set_state(state)
    return palette
