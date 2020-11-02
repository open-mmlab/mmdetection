import math

import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyNewDataset(CustomDataset):
    CLASSES = ('ROI')

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            anns = ann_line.split(' ')
            bboxes = []
            labels = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                anns = anns.split(' ')

                anns[0] = float(anns[0])
                anns[1] = float(anns[1])

                #Coco -> YOLO
                anns[2] = float(anns[0]) + float(anns[2])
                anns[3] = float(anns[1]) + float(anns[3])

                #PASCAL VOC -> YOLO
                # anns[2] = float(anns[2]) - float(anns[0])
                # anns[3] = float(anns[3]) - float(anns[1])

                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))

            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']