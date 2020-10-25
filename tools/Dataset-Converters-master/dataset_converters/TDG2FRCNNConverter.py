from dataset_converters.ConverterBase import ConverterBase

import os
import cv2


class TDG2FRCNNConverter(ConverterBase):

    formats = ['TDG2FRCNN']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _run(self, input_folder, output_folder, FORMAT):

        f = open(os.path.join(input_folder, 'bboxes.txt'), 'r')

        text = f.read()
        f.close()

        saveDir = output_folder
        imgDir = os.path.join(saveDir, 'Images')

        self._ensure_folder_exists_and_is_clear(saveDir)
        self._ensure_folder_exists_and_is_clear(imgDir)

        lines = text.split('\n')
        assert len(lines) > 0

        saveCount = 1
        i = 0
        lastName = None
        lastClass = None

        f = open(os.path.join(saveDir, 'loc.trainval'), 'w')

        for line in lines:

            rois = []

            tokens = line.split(' ')

            assert len(tokens) > 1 and ((len(tokens) - 1) % 5 == 0)

            name = os.path.join(imgDir, str(saveCount) + '.jpg')

            for token in tokens:
                if '.bmp' in token or '.jpg' in token or '.png' in token:
                    lastName = token
                    self.copy(os.path.join(input_folder, token), name)
                else:

                    if i == 0:
                        lastClass = int(token)
                        assert(lastClass >= 1)
                    elif i == 1:
                        x = int(token)
                        assert x >= 0
                    elif i == 2:
                        y = int(token)
                        assert y >= 0
                    elif i == 3:
                        w = int(token)
                        assert w >= 2
                    elif i == 4:
                        h = int(token)
                        assert h >= 2

                    i += 1

                    if i % 5 == 0:
                        i = 0
                        rois.append([lastClass, x, y, x + w - 1, y + h - 1])

            f.write('# ' + str(saveCount - 1) + '\n')
            f.write('Images/' + str(saveCount) + '.jpg\n')
            f.write(str(len(rois)) + '\n')
            for roi in rois:
                f.write(str(roi[0]))
                for coord in roi[1:]:
                    f.write(' ' + str(coord))
                f.write(' 0\n')
            f.write('\n')

            saveCount += 1
