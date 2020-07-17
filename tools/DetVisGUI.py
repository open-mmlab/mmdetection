# __author__ = 'ChienHung Chen in Academia Sinica IIS'

import argparse
import itertools
import json
import os
import pickle
import xml.etree.ElementTree as ET
from tkinter import (END, Button, Checkbutton, E, Entry, IntVar, Label,
                     Listbox, Menu, N, S, Scrollbar, StringVar, Tk, W, ttk)

import cv2
import matplotlib
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image, ImageTk

matplotlib.use('TkAgg')


def parse_args():
    parser = argparse.ArgumentParser(description='DetVisGUI')

    parser.add_argument('config', help='config file path')
    parser.add_argument('det_file', help='detection results file path')

    parser.add_argument(
        '--stage',
        default='test',
        choices=['train', 'val', 'test'],
        help='stage')

    parser.add_argument(
        '--no_gt',
        action='store_true',
        default=False,
        help='test images without groundtruth')

    parser.add_argument(
        '--det_box_color', default=(255, 255, 0), help='detection box color')

    parser.add_argument(
        '--gt_box_color',
        default=(255, 255, 255),
        help='groundtruth box color')

    parser.add_argument('--output', default='', help='image save folder')

    args = parser.parse_args()
    return args


class COCO_dataset:

    def __init__(self, cfg, args):
        self.dataset = 'COCO'
        self.img_root = getattr(cfg.data, args.stage).img_prefix
        self.anno_root = getattr(cfg.data, args.stage).ann_file
        self.det_file = args.det_file
        self.has_anno = not args.no_gt
        self.mask = False

        # according json to get category, image list, and annotations.
        self.category, self.img_list, self.total_annotations = self.parse_json(
            self.anno_root, self.has_anno)
        self.aug_category = aug_category(self.category)

        self.results = self.get_det_results() if self.det_file != '' else None

        if self.det_file != '':
            self.img_det = {
                self.img_list[i]: self.results[:, i]
                for i in range(len(self.img_list))
            }

    def parse_json(self, train_anno, has_anno):
        with open(train_anno) as f:
            data = json.load(f)

        category = [c['name'] for c in data['categories']]  # 80 classes

        if has_anno:
            annotations = data['annotations']
        images = data['images']

        category_dict = {c['id']: c['name'] for c in data['categories']}
        max_category_id = max(category_dict.keys())

        # id to image mapping
        image_dict = {}
        img_list = list()

        for image in images:
            key = image['id']
            image_dict[key] = [
                image['file_name'], image['width'], image['height']
            ]
            img_list.append(image['file_name'])

        category_count = [0 for _ in range(max_category_id)]

        total_annotations = {}

        if has_anno:
            for a in annotations:
                image_name = image_dict[a['image_id']][0].replace('.jpg', '')
                width = image_dict[a['image_id']][1]
                height = image_dict[a['image_id']][2]
                idx = a['category_id']
                single_ann = []
                single_ann.append(category_dict[idx])
                single_ann.extend(list(map(int, a['bbox'])))
                single_ann.extend([width, height])

                if image_name not in total_annotations:
                    total_annotations[image_name] = []

                category_count[idx - 1] += 1
                total_annotations[image_name].append(single_ann)

            print('\n==============[ {} json info ]=============='.format(
                self.dataset))
            print('Total Annotations: {}'.format(len(annotations)))
            print('Total Image      : {}'.format(len(images)))
            print('Annotated Image  : {}'.format(len(total_annotations)))
            print('Total Category   : {}'.format(len(category)))
            print('----------------------------')
            print('{:^20}| count'.format('class'))
            print('----------------------------')
            for c, cnt in zip(category, category_count):
                if cnt != 0:
                    print('{:^20}| {}'.format(c, cnt))
            print()
        return category, img_list, total_annotations

    def get_det_results(self):
        det_file = self.det_file
        if det_file != '':
            with open(det_file, 'rb') as f:
                det_results = np.asarray(
                    pickle.load(f))  # [(bg + cls), images]

            # dim should be (class, image), mmdetection format: (image, class)
            if len(det_results.shape) == 2:
                det_results = np.transpose(det_results, (1, 0))
            elif len(det_results.shape) == 3:
                det_results = np.transpose(det_results, (2, 0, 1))
                self.mask = True

            return det_results

        else:
            return None

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root,
                                      self.img_list[idx])).convert('RGB')
        return img

    def get_singleImg_gt(self, name):
        if name.replace('.jpg', '') not in self.total_annotations.keys():
            print('There are no annotations in %s.' % name)
            return []
        else:
            return self.total_annotations[name.replace('.jpg', '')]

    def get_singleImg_dets(self, name):
        return self.img_det[name]


# pascal voc dataset
class VOC_dataset:

    def __init__(self, cfg, args):
        self.dataset = 'PASCAL VOC'
        self.det_file = args.det_file
        self.txt = getattr(cfg.data, args.stage).ann_file
        self.anno_root = os.path.join(
            getattr(cfg.data, args.stage).img_prefix, 'Annotations')
        self.img_root = os.path.join(
            getattr(cfg.data, args.stage).img_prefix, 'JPEGImages')
        self.has_anno = not args.no_gt
        self.mask = False

        # according txt to get image list
        self.img_list = self.get_img_list()

        self.results = self.get_det_results() if self.det_file != '' else None
        self.aug_category = aug_category([
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ])

        if self.det_file != '':
            self.img_det = {
                self.img_list[i]: self.results[:, i]
                for i in range(len(self.img_list))
            }

    def get_img_list(self):
        with open(self.txt, 'r') as f:
            data = f.readlines()

        return [x.strip() + '.jpg' for x in data]

    def get_det_results(self):
        det_file = self.det_file
        if det_file != '':
            with open(det_file, 'rb') as f:
                det_results = np.asarray(
                    pickle.load(f))  # [(bg + cls), images]

            # dim should be (class, image), mmdetection format: (image, class)
            det_results = np.transpose(det_results, (1, 0))

            return det_results

        else:
            return None

    def get_img_by_name(self, name):
        img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        return img

    def get_img_by_index(self, idx):
        img = Image.open(os.path.join(self.img_root,
                                      self.img_list[idx])).convert('RGB')
        return img

    def get_singleImg_gt(self, name):  # get annotations by image name
        xml_path = os.path.join(self.anno_root,
                                name.replace('.jpg', '') + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_anns = []
        for obj in root.findall('object'):
            single_ann = []
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text),
                int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            ]
            single_ann.append(name)
            single_ann.extend(bbox)
            img_anns.append(single_ann)

        return img_anns

    def get_singleImg_dets(self, name):
        return self.img_det[name]


# main GUI
class vis_tool:

    def __init__(self):
        self.args = parse_args()
        cfg = mmcv.Config.fromfile(self.args.config)
        self.window = Tk()
        self.menubar = Menu(self.window)

        self.info = StringVar()
        self.info_label = Label(
            self.window, bg='yellow', width=4, textvariable=self.info)

        self.listBox_img = Listbox(
            self.window, width=50, height=20, font=('Times New Roman', 10))
        self.listBox_obj = Listbox(
            self.window, width=50, font=('Times New Roman', 10))

        self.scrollbar_img = Scrollbar(
            self.window, width=15, orient='vertical')
        self.scrollbar_obj = Scrollbar(
            self.window, width=15, orient='vertical')

        self.listBox_img_info = StringVar()
        self.listBox_img_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_img_info)

        self.listBox_obj_info = StringVar()
        self.listBox_obj_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=4,
            height=1,
            textvariable=self.listBox_obj_info)

        if cfg.dataset_type == 'VOCDataset':
            self.data_info = VOC_dataset(cfg, self.args)
        elif cfg.dataset_type == 'CocoDataset':
            self.data_info = COCO_dataset(cfg, self.args)

        self.info.set('DATASET: {}'.format(self.data_info.dataset))

        # load image and show it on the window
        self.img = self.data_info.get_img_by_index(0)
        self.photo = ImageTk.PhotoImage(self.img)
        self.label_img = Label(self.window, image=self.photo)

        self.show_txt = IntVar(value=1)
        self.checkbn_txt = Checkbutton(
            self.window,
            text='LabelText',
            font=('Arial', 10, 'bold'),
            variable=self.show_txt,
            command=self.change_img)

        self.show_dets = IntVar(value=1)
        self.checkbn_det = Checkbutton(
            self.window,
            text='Detections',
            font=('Arial', 10, 'bold'),
            variable=self.show_dets,
            command=self.change_img,
            fg='#0000FF')

        self.show_gts = IntVar(value=1)
        self.checkbn_gt = Checkbutton(
            self.window,
            text='Groundtruth',
            font=('Arial', 10, 'bold'),
            variable=self.show_gts,
            command=self.change_img,
            fg='#FF8C00')

        self.combo_label = Label(
            self.window,
            bg='yellow',
            width=10,
            height=1,
            text='Show Category',
            font=('Arial', 11))
        self.combo_category = ttk.Combobox(
            self.window,
            font=('Arial', 11),
            values=self.data_info.aug_category.combo_list)
        self.combo_category.current(0)

        self.th_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='Threshold')
        self.threshold = np.float32(0.5)
        self.th_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.threshold)))
        self.th_button = Button(
            self.window, text='Enter', height=1, command=self.change_threshold)

        self.find_label = Label(
            self.window,
            font=('Arial', 11),
            bg='yellow',
            width=10,
            height=1,
            text='find')
        self.find_name = ''
        self.find_entry = Entry(
            self.window,
            font=('Arial', 11),
            width=10,
            textvariable=StringVar(self.window, value=str(self.find_name)))
        self.find_button = Button(
            self.window, text='Enter', height=1, command=self.findname)

        self.listBox_img_idx = 0

        # ====== ohter attribute ======
        self.img_name = ''
        self.show_img = None

        if not self.args.output:
            self.output = os.path.join(cfg.work_dir, 'output')

        if not os.path.isdir(self.output):
            os.makedirs(self.output)

        self.img_list = self.data_info.img_list

        # flag for find/threshold button switch focused element
        self.button_clicked = False

    def change_threshold(self, event=None):
        try:
            self.threshold = np.float32(self.th_entry.get())
            self.change_img()

            # after changing threshold, focus on listBox for easy control
            if self.window.focus_get() == self.listBox_obj:
                self.listBox_obj.focus()
            else:
                self.listBox_img.focus()

            self.button_clicked = True

        except ValueError:
            self.window.title('Please enter a number as score threshold.')

    # draw groundtruth
    def draw_gt_boxes(self, img, objs):
        for obj in objs:
            cls_name = obj[0]

            # according combobox to decide whether to plot this category
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if cls_name not in show_category:
                continue

            box = obj[1:]
            xmin = max(box[0], 0)
            ymin = max(box[1], 0)
            xmax = min(box[0] + box[2], self.img_width)
            ymax = min(box[1] + box[3], self.img_height)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if self.show_txt.get():
                if ymax + 30 >= self.img_height:
                    cv2.rectangle(img, (xmin, ymin),
                                  (xmin + len(cls_name) * 10, int(ymin - 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymin - 5)), font,
                                0.5, (255, 255, 255), 1)
                else:
                    cv2.rectangle(img, (xmin, ymax),
                                  (xmin + len(cls_name) * 10, int(ymax + 20)),
                                  (255, 140, 0), cv2.FILLED)
                    cv2.putText(img, cls_name, (xmin, int(ymax + 15)), font,
                                0.5, (255, 255, 255), 1)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                          self.args.gt_box_color, 1)

        return img

    def draw_all_det_boxes(self, img, single_detection):
        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj in cls_objs:
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if self.show_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(
                                img, (xmin, ymin),
                                (xmin + len(text) * 9, int(ymin - 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                                        0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(
                                img, (xmin, ymax),
                                (xmin + len(text) * 9, int(ymax + 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)),
                                        font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                  self.args.det_box_color, 2)

        return img

    def draw_all_det_boxes_masks(self, img, single_detection):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])
            category = self.data_info.aug_category.category[show_idx]

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for i in inds:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
            self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        # draw bounding box
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj in cls_objs:
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    box = list(map(int, list(map(round, box))))
                    xmin = max(box[0], 0)
                    ymin = max(box[1], 0)
                    xmax = min(box[2], self.img_width)
                    ymax = min(box[3], self.img_height)

                    if self.show_txt.get():
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = category + ' : ' + str(score)

                        if ymax + 30 >= self.img_height:
                            cv2.rectangle(
                                img, (xmin, ymin),
                                (xmin + len(text) * 9, int(ymin - 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymin - 5)), font,
                                        0.5, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(
                                img, (xmin, ymax),
                                (xmin + len(text) * 9, int(ymax + 20)),
                                (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, text, (xmin, int(ymax + 15)),
                                        font, 0.5, (255, 255, 255), 1)

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                  self.args.det_box_color, 2)

        return img

    def change_img(self, event=None):
        if len(self.listBox_img.curselection()) != 0:
            self.listBox_img_idx = self.listBox_img.curselection()[0]

        self.listBox_img_info.set('Image  {:6}  / {:6}'.format(
            self.listBox_img_idx + 1, self.listBox_img.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        self.window.title('DATASET : ' + self.data_info.dataset + '   ' + name)

        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height

        img = np.asarray(img)

        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():
            if self.data_info.mask is False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_all_det_boxes(img, dets)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose(
                    (1, 0))
                img = self.draw_all_det_boxes_masks(img, dets)

            self.clear_add_listBox_obj()

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    def draw_one_det_boxes(self, img, single_detection, selected_idx=-1):
        idx_counter = 0
        for idx, cls_objs in enumerate(single_detection):

            category = self.data_info.aug_category.category[idx]
            if self.combo_category.get() == 'All':
                show_category = self.data_info.aug_category.category
            else:
                show_category = [self.combo_category.get()]

            if category not in show_category:
                continue

            for obj in cls_objs:
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if self.show_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                      self.args.det_box_color, 2)

                        return img
                    else:
                        idx_counter += 1

    def draw_one_det_boxes_masks(self, img, single_detection, selected_idx=-1):
        img = np.require(img, requirements=['W'])
        boxes, masks = single_detection

        # draw segmentation masks
        # reference mmdetection/mmdet/models/detectors/base.py
        if self.combo_category.get() != 'All':
            show_idx = self.data_info.aug_category.category.index(
                self.combo_category.get())
            category = self.data_info.aug_category.category[
                show_idx]  # fixed category
            masks = np.asarray([masks[show_idx]])
            boxes = np.asarray([boxes[show_idx]])

        segms = list(itertools.chain(*masks))
        bboxes = np.vstack(boxes)

        inds = np.where(np.round(bboxes[:, -1], 2) >= self.threshold)[0]

        self.color_list = []
        for inds_idx, i in enumerate(inds):
            if inds_idx == selected_idx:
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
                self.color_list.append('#%02x%02x%02x' % tuple(color_mask[0]))

        # draw bounding box
        idx_counter = 0
        for idx, cls_objs in enumerate(boxes):
            if self.combo_category.get() == 'All':
                category = self.data_info.aug_category.category[idx]

            for obj_idx, obj in enumerate(cls_objs):
                [score, box] = [round(obj[4], 2), obj[:4]]

                if score >= self.threshold:
                    if idx_counter == selected_idx:
                        box = list(map(int, list(map(round, box))))
                        xmin = max(box[0], 0)
                        ymin = max(box[1], 0)
                        xmax = min(box[2], self.img_width)
                        ymax = min(box[3], self.img_height)

                        if self.show_txt.get():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text = category + ' : ' + str(score)

                            if ymax + 30 >= self.img_height:
                                cv2.rectangle(
                                    img, (xmin, ymin),
                                    (xmin + len(text) * 9, int(ymin - 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymin - 5)),
                                            font, 0.5, (255, 255, 255), 1)
                            else:
                                cv2.rectangle(
                                    img, (xmin, ymax),
                                    (xmin + len(text) * 9, int(ymax + 20)),
                                    (0, 0, 255), cv2.FILLED)
                                cv2.putText(img, text, (xmin, int(ymax + 15)),
                                            font, 0.5, (255, 255, 255), 1)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                                      self.args.det_box_color, 2)

                        return img
                    else:
                        idx_counter += 1

    # plot only one object
    def change_obj(self, event=None):
        if len(self.listBox_obj.curselection()) == 0:
            self.listBox_img.focus()
            return
        else:
            listBox_obj_idx = self.listBox_obj.curselection()[0]

        self.listBox_obj_info.set('Detected Object : {:4}  / {:4}'.format(
            listBox_obj_idx + 1, self.listBox_obj.size()))

        name = self.listBox_img.get(self.listBox_img_idx)
        img = self.data_info.get_img_by_name(name)
        self.img_width, self.img_height = img.width, img.height
        img = np.asarray(img)
        self.img_name = name
        self.img = img

        if self.data_info.has_anno and self.show_gts.get():
            objs = self.data_info.get_singleImg_gt(name)
            img = self.draw_gt_boxes(img, objs)

        if self.data_info.results is not None and self.show_dets.get():

            if self.data_info.mask is False:
                dets = self.data_info.get_singleImg_dets(name)
                img = self.draw_one_det_boxes(img, dets, listBox_obj_idx)
            else:
                dets = self.data_info.get_singleImg_dets(name).transpose(
                    (1, 0))
                img = self.draw_one_det_boxes_masks(img, dets, listBox_obj_idx)

        self.show_img = img
        img = Image.fromarray(img)
        img = self.scale_img(img)
        self.photo = ImageTk.PhotoImage(img)
        self.label_img.config(image=self.photo)
        self.window.update_idletasks()

        if self.img_name in os.listdir(self.output):
            self.listBox_img_label.config(bg='#CCFF99')
        else:
            self.listBox_img_label.config(bg='yellow')

    # ============================================

    def scale_img(self, img):
        [s_w, s_h] = [1, 1]

        # if window size is (1920, 1080),
        # the default max image size is (1440, 810)
        (fix_width, fix_height) = (1440, 810)

        # change image size according to window size
        if self.window.winfo_width() != 1:
            fix_width = (
                self.window.winfo_width() - self.listBox_img.winfo_width() -
                self.scrollbar_img.winfo_width() - 5)
            fix_height = int(fix_width * 9 / 16)

        # handle image size is too big
        if img.width > fix_width:
            s_w = fix_width / img.width
        if img.height > fix_height:
            s_h = fix_height / img.height

        scale = min(s_w, s_h)
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.ANTIALIAS)
        return img

    def clear_add_listBox_obj(self):
        self.listBox_obj.delete(0, 'end')

        if self.data_info.mask is False:
            single_detection = self.data_info.get_singleImg_dets(
                self.img_list[self.listBox_img_idx])
        else:
            single_detection, single_mask = self.data_info.get_singleImg_dets(
                self.img_list[self.listBox_img_idx]).transpose((1, 0))

        if self.combo_category.get() == 'All':
            show_category = self.data_info.aug_category.category
        else:
            show_category = [self.combo_category.get()]

        num = 0
        for idx, cls_objs in enumerate(single_detection):
            category = self.data_info.aug_category.category[idx]

            if category not in show_category:
                continue

            for obj in cls_objs:
                score = np.round(obj[4], 2)
                if score >= self.threshold:
                    self.listBox_obj.insert('end',
                                            category + ' : ' + str(score))
                    num += 1

        self.listBox_obj_info.set('Detected Object : {:3}'.format(num))

    def change_threshold_button(self, v):
        self.threshold += v

        if self.threshold <= 0:
            self.threshold = 0
        elif self.threshold >= 1:
            self.threshold = 1

        self.th_entry.delete(0, END)
        self.th_entry.insert(0, str(round(self.threshold, 2)))
        self.change_threshold()

    def save_img(self):
        print('Save image to ' + os.path.join(self.output, self.img_name))
        cv2.imwrite(
            os.path.join(self.output, self.img_name),
            cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB))
        self.listBox_img_label.config(bg='#CCFF99')

    def eventhandler(self, event):
        if self.window.focus_get() not in [self.find_entry, self.th_entry]:
            if event.keysym == 'Right':
                self.change_threshold_button(0.1)
            elif event.keysym == 'Left':
                self.change_threshold_button(-0.1)
            elif event.keysym == 'q':
                self.window.quit()
            elif event.keysym == 's':
                self.save_img()

            if self.button_clicked:
                self.button_clicked = False
            else:
                if event.keysym in ['KP_Enter', 'Return']:
                    self.listBox_obj.focus()
                    self.listBox_obj.select_set(0)
                elif event.keysym == 'Escape':
                    self.change_img()
                    self.listBox_img.focus()

    def combobox_change(self, event=None):
        self.listBox_img.focus()
        self.change_img()

    def clear_add_listBox_img(self):
        self.listBox_img.delete(0, 'end')  # delete listBox_img 0 ~ end items

        # add image name to listBox_img
        for item in self.img_list:
            self.listBox_img.insert('end', item)

        self.listBox_img.select_set(0)
        self.listBox_img.focus()
        self.change_img()

    def findname(self, event=None):
        self.find_name = self.find_entry.get()
        new_list = []

        if self.find_name == '':
            new_list = self.data_info.img_list
        else:
            for img_name in self.data_info.img_list:
                if self.find_name[0] == '!':
                    if self.find_name[1:] not in img_name:
                        new_list.append(img_name)
                else:
                    if self.find_name in img_name:
                        new_list.append(img_name)

        if len(new_list) != 0:
            self.img_list = new_list
            self.clear_add_listBox_img()
            self.clear_add_listBox_obj()
            self.button_clicked = True
        else:
            self.window.title("Can't find any image about '{}'".format(
                self.find_name))

    def run(self):
        self.window.title('DATASET : ' + self.data_info.dataset)
        self.window.geometry('1280x800+350+100')

        # self.menubar.add_command(label='QUIT', command=self.window.quit)
        self.window.config(menu=self.menubar)  # display the menu
        self.scrollbar_img.config(command=self.listBox_img.yview)
        self.listBox_img.config(yscrollcommand=self.scrollbar_img.set)
        self.scrollbar_obj.config(command=self.listBox_obj.yview)
        self.listBox_obj.config(yscrollcommand=self.scrollbar_obj.set)

        layer1 = 0
        layer2 = 50

        # ======================= layer 1 =========================

        # combobox
        self.combo_label.grid(
            row=layer1 + 30,
            column=0,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)
        self.combo_category.grid(
            row=layer1 + 30,
            column=6,
            sticky=W + E + N + S,
            padx=3,
            pady=3,
            columnspan=6)

        # show label
        self.checkbn_det.grid(
            row=layer1 + 40,
            column=0,
            sticky=N + S,
            padx=3,
            pady=3,
            columnspan=4)
        # show gt
        self.checkbn_gt.grid(
            row=layer1 + 40,
            column=4,
            sticky=N + S,
            padx=3,
            pady=3,
            columnspan=4)
        # show det
        self.checkbn_txt.grid(
            row=layer1 + 40,
            column=8,
            sticky=N + S,
            padx=3,
            pady=3,
            columnspan=4)

        # ======================= layer 2 =========================

        self.listBox_img_label.grid(
            row=layer2 + 0, column=0, sticky=N + S + E + W, columnspan=12)

        # find name
        self.find_label.grid(
            row=layer2 + 20, column=0, sticky=E + W, columnspan=4)
        self.find_entry.grid(
            row=layer2 + 20, column=4, sticky=E + W, columnspan=4)
        self.find_button.grid(
            row=layer2 + 20, column=8, sticky=E + W, pady=3, columnspan=4)

        self.scrollbar_img.grid(row=layer2 + 30, column=11, sticky=N + S + W)
        self.label_img.grid(
            row=layer1 + 30,
            column=12,
            sticky=N + E,
            padx=3,
            pady=3,
            rowspan=110)
        self.listBox_img.grid(
            row=layer2 + 30,
            column=0,
            sticky=N + S + E + W,
            pady=3,
            columnspan=11)

        if self.data_info.det_file != '':
            self.th_label.grid(
                row=layer2 + 40, column=0, sticky=E + W, columnspan=4)
            self.th_entry.grid(
                row=layer2 + 40, column=4, sticky=E + W, columnspan=4)
            self.th_button.grid(
                row=layer2 + 40, column=8, sticky=E + W, pady=3, columnspan=4)

            self.listBox_obj_label.grid(
                row=layer2 + 50, column=0, sticky=E + W, pady=3, columnspan=12)

            self.scrollbar_obj.grid(
                row=layer2 + 60, column=11, sticky=N + S + W, pady=3)
            self.listBox_obj.grid(
                row=layer2 + 60,
                column=0,
                sticky=N + S + E + W,
                pady=3,
                columnspan=11)

        self.clear_add_listBox_img()
        self.listBox_img.bind('<<ListboxSelect>>', self.change_img)
        self.listBox_img.bind_all('<KeyRelease>', self.eventhandler)

        self.listBox_obj.bind('<<ListboxSelect>>', self.change_obj)

        self.th_entry.bind('<Return>', self.change_threshold)
        self.th_entry.bind('<KP_Enter>', self.change_threshold)
        self.find_entry.bind('<Return>', self.findname)
        self.find_entry.bind('<KP_Enter>', self.findname)

        self.combo_category.bind('<<ComboboxSelected>>', self.combobox_change)

        self.window.mainloop()


class aug_category:

    def __init__(self, categories):
        self.category = categories
        self.combo_list = categories.copy()
        self.combo_list.insert(0, 'All')
        self.all = True


if __name__ == '__main__':
    vis_tool().run()
