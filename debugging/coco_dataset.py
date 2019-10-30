"""Coco Dataset.

Coco Dataset without any images with crowds. Done temporarily to prevent issues
with the transformation.
"""
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, root, annFile, transforms=None):
        self.coco = CocoDetection(root, annFile)
        self.transforms = transforms
        self.length = len(self.coco)

    def __getitem__(self, index: int):
        is_crowd = True
        while is_crowd:
            is_crowd = False
            img, annotations = self.coco[index]
            for annotation in annotations:
                if annotation['iscrowd'] == 1:
                    is_crowd = True
                    index += 1
                    if index >= self.length:
                        index = 0

        if self.transforms is not None:
            img, annotations = self.transforms(img, annotations)

        return img, annotations

    def __len__(self):
        return self.length