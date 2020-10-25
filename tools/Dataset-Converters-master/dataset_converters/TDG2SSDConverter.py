from dataset_converters.ConverterBase import ConverterBase

import os
import cv2


class TDG2SSDConverter(ConverterBase):

    formats = ['TDG2SSD']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _run(self, input_folder, output_folder, FORMAT):

        f = open(os.path.join(input_folder, 'bboxes.txt'), 'r')

        text = f.read()
        f.close()

        SSDDir = output_folder
        DBDir = os.path.join(SSDDir, 'DB')
        ImagesDir = os.path.join(DBDir, 'Images')
        AnnotationDir = os.path.join(DBDir, 'Annotations')

        self._ensure_folder_exists_and_is_clear(SSDDir)
        self._ensure_folder_exists_and_is_clear(DBDir)
        self._ensure_folder_exists_and_is_clear(ImagesDir)
        self._ensure_folder_exists_and_is_clear(AnnotationDir)

        infile = open(os.path.join(SSDDir, 'infile.txt'), 'w')

        lines = text.split('\n')
        assert len(lines) > 0

        saveCount = 1
        i = 0
        lastName = None
        lastClass = None

        for line in lines:

            if len(line) == 0:
                continue

            annotation = open(os.path.join(AnnotationDir, str(saveCount) + '.txt'), 'w')

            tokens = line.split(' ')

            assert len(tokens) > 1 and ((len(tokens) - 1) % 5 == 0)

            for token in tokens:
                if '.bmp' in token or '.jpg' in token or '.png' in token:
                    lastName = token
                    self.copy(os.path.join(input_folder, token), os.path.join(ImagesDir, token))
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
                        annotation.write('{} {} {} {} {}\n'.format(lastClass, x, y, x + w - 1, y + h - 1))

            infile.write('Images/' + lastName + ' Annotations/' + str(saveCount) + '.txt\n')
            saveCount += 1
