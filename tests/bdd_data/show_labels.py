import json
import argparse
import urllib
from multiprocessing import Pool
import os
from os.path import exists, splitext, isdir, isfile, join, split, dirname
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from PIL import Image
import sys
from collections import Iterable
import io

from .label import labels
from .geometry import Label3d

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2018, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'BSD'


def parse_args():
    """Use argparse to get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=False,
                        help="input raw image", type=str)
    parser.add_argument('--image-dir', help='image directory')
    parser.add_argument("-l", "--label", required=True,
                        help="corresponding bounding box annotation "
                             "(json file)", type=str)
    parser.add_argument('-s', '--scale', type=int, default=1,
                        help="Scale up factor for annotation factor. "
                             "Useful when producing visualization as "
                             "thumbnails.")
    parser.add_argument('--no-attr', action='store_true', default=False,
                        help="Do not show attributes")
    parser.add_argument('--no-lane', action='store_true', default=False,
                        help="Do not show lanes")
    parser.add_argument('--no-drivable', action='store_true', default=False,
                        help="Do not show drivable areas")
    parser.add_argument('--no-box2d', action='store_true', default=False,
                        help="Do not show 2D bounding boxes")
    parser.add_argument("-o", "--output_dir", required=False, default=None,
                        type=str,
                        help="output image file with bbox visualization. "
                             "If it is set, the images will be written to the "
                             "output folder instead of being displayed "
                             "interactively.")
    parser.add_argument('--instance', action='store_true', default=False,
                        help='Set instance segmentation mode')
    parser.add_argument('--drivable', action='store_true', default=False,
                        help='Set drivable area mode')
    parser.add_argument('--target-objects', type=str, default='',
                        help='A comma separated list of objects. If this is '
                             'not empty, only show images with the target '
                             'objects.')
    parser.add_argument('--format', default='v2')
    args = parser.parse_args()

    # Check if the corresponding bounding box annotation exits
    # is_valid_file(parser, args.image)
    # is_valid_file(parser, args.label)
    # assert (isdir(args.image) and isdir(args.label)) or \
    #        (isfile(args.image) and isfile(args.label)), \
    #     "input and label should be both folders or files"
    if len(args.target_objects) > 0:
        args.target_objects = args.target_objects.split(',')

    return args


def is_valid_file(parser, file_name):
    """Ensure that the file exists."""

    if not exists(file_name):
        parser.error("The corresponding bounding box annotation '{}' does "
                     "not exist!".format(file_name))
        sys.exit(1)


def get_areas_v0(objects):
    return [o for o in objects
            if 'poly2d' in o and o['category'][:4] == 'area']


def get_areas(objects):
    return [o for o in objects
            if 'poly2d' in o and o['category'] == 'drivable area']


def get_lanes(objects):
    return [o for o in objects
            if 'poly2d' in o and o['category'][:4] == 'lane']


def get_other_poly2d(objects):
    return [o for o in objects
            if 'poly2d' in o and o['poly2d'] is not None and
            (o['category'] not in ['drivable area', 'lane'])]


def get_boxes(objects):
    return [o for o in objects if ('box2d' in o and o['box2d'] is not None)
            or ('box3d' in o and o['box3d'] is not None)]


def get_target_objects(objects, targets):
    return [o for o in objects if o['category'] in targets]


def random_color():
    return np.random.rand(3)


def seg2color(seg):
    num_ids = 20
    train_colors = np.zeros((num_ids, 3), dtype=np.uint8)
    for l in labels:
        if l.trainId < 255:
            train_colors[l.trainId] = l.color
    color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for i in range(num_ids):
        color[seg == i, :] = train_colors[i]
    return color


def instance2color(instance):
    instance_colors = dict([(i, (np.random.random(3) * 255).astype(np.uint8))
                            for i in np.unique(instance)])
    color = np.zeros((instance.shape[0], instance.shape[1], 3), dtype=np.uint8)
    for k, v in instance_colors.items():
        color[instance == k] = v
    return color


def convert_instance_rgb(label_path):
    label_dir = dirname(label_path)
    label_name = splitext(split(label_path)[1])[0]
    image = np.array(Image.open(label_path, 'r'))
    seg = image[:, :, 0]
    seg_color = seg2color(seg)
    image = image.astype(np.uint32)
    instance = image[:, :, 0] * 1000 + image[:, :, 1]
    # instance_color = instance2color(instance)
    Image.fromarray(seg).save(
        join(label_dir, label_name + '_train_id.png'))
    Image.fromarray(seg_color).save(
        join(label_dir, label_name + '_train_color.png'))
    Image.fromarray(instance).save(
        join(label_dir, label_name + '_instance_id.png'))
    # Image.fromarray(instance_color).save(
    #     join(label_dir, label_name + '_instance_color.png'))


def drivable2color(seg):
    colors = [[0, 0, 0, 255],
              [217, 83, 79, 255],
              [91, 192, 222, 255]]
    color = np.zeros((seg.shape[0], seg.shape[1], 4), dtype=np.uint8)
    for i in range(3):
        color[seg == i, :] = colors[i]
    return color


def convert_drivable_rgb(label_path):
    label_dir = dirname(label_path)
    label_name = splitext(split(label_path)[1])[0]
    image = np.array(Image.open(label_path, 'r'))
    seg = image[:, :, 0]
    seg_color = drivable2color(seg)
    image = image.astype(np.uint32)
    instance = image[:, :, 0] * 1000 + image[:, :, 2]
    # instance_color = instance2color(instance)
    Image.fromarray(seg).save(
        join(label_dir, label_name + '_drivable_id.png'))
    Image.fromarray(seg_color).save(
        join(label_dir, label_name + '_drivable_color.png'))
    Image.fromarray(instance).save(
        join(label_dir, label_name + '_drivable_instance_id.png'))
    # Image.fromarray(instance_color).save(
    #     join(label_dir, label_name + '_drivable_instance_color.png'))


class LabelViewer(object):
    def __init__(self, args):
        """Visualize bounding boxes"""
        self.ax = None
        self.fig = None
        self.current_index = 0
        self.scale = args.scale
        image_paths = [args.image]
        label_paths = [args.label]
        if isdir(args.label):
            input_names = sorted(
                [splitext(n)[0] for n in os.listdir(args.label)
                 if splitext(n)[1] == '.json'])
            image_paths = [join(args.image, n + '.jpg') for n in input_names]
            label_paths = [join(args.label, n + '.json') for n in input_names]
        self.image_paths = image_paths
        self.label_paths = label_paths

        self.font = FontProperties()
        self.font.set_family(['Luxi Mono', 'monospace'])
        self.font.set_weight('bold')
        self.font.set_size(18 * self.scale)

        self.with_image = True
        self.with_attr = not args.no_attr
        self.with_lane = not args.no_lane
        self.with_drivable = not args.no_drivable
        self.with_box2d = not args.no_box2d
        self.with_segment = True

        self.target_objects = args.target_objects

        if len(self.target_objects) > 0:
            print('Only showing objects:', self.target_objects)

        self.out_dir = args.output_dir
        self.label_map = dict([(l.name, l) for l in labels])
        self.color_mode = 'random'

        self.image_width = 1280
        self.image_height = 720

        self.instance_mode = False
        self.drivable_mode = False
        self.with_post = False  # with post processing

        if args.drivable:
            self.set_drivable_mode()

        if args.instance:
            self.set_instance_mode()

    def view(self):
        self.current_index = 0
        if self.out_dir is None:
            self.show()
        else:
            self.write()

    def show(self):
        # Read and draw image
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        if len(self.image_paths) > 1:
            plt.connect('key_release_event', self.next_image)
        self.show_image()
        plt.show()

    def write(self):
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        out_paths = []
        for i in range(len(self.image_paths)):
            self.current_index = i
            out_name = splitext(split(self.image_paths[i])[1])[0] + '.png'
            out_path = join(self.out_dir, out_name)
            if self.show_image():
                self.fig.savefig(out_path, dpi=dpi)
                out_paths.append(out_path)
        if self.with_post:
            print('Post-processing')
            p = Pool(10)
            if self.instance_mode:
                p.map(convert_instance_rgb, out_paths)
            if self.drivable_mode:
                p = Pool(10)
                p.map(convert_drivable_rgb, out_paths)

    def set_instance_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = False
        self.with_lane = False
        self.with_box2d = False
        self.with_segment = True
        self.color_mode = 'instance'
        self.instance_mode = True
        self.with_post = True

    def set_drivable_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = True
        self.with_lane = False
        self.with_box2d = False
        self.with_segment = False
        self.color_mode = 'instance'
        self.drivable_mode = True
        self.with_post = True

    def show_image(self):
        plt.cla()
        label_path = self.label_paths[self.current_index]
        name = splitext(split(label_path)[1])[0]
        print('Image:', name)
        self.fig.canvas.set_window_title(name)

        if self.with_image:
            image_path = self.image_paths[self.current_index]
            img = mpimg.imread(image_path)
            im = np.array(img, dtype=np.uint8)
            self.ax.imshow(im, interpolation='nearest', aspect='auto')
        else:
            self.ax.set_xlim(0, self.image_width - 1)
            self.ax.set_ylim(0, self.image_height - 1)
            self.ax.invert_yaxis()
            self.ax.add_patch(self.poly2patch(
                [[0, 0, 'L'], [0, self.image_height - 1, 'L'],
                 [self.image_width - 1, self.image_height - 1, 'L'],
                 [self.image_width - 1, 0, 'L']],
                closed=True, alpha=1., color='black'))

        # Read annotation labels
        with open(label_path) as data_file:
            label = json.load(data_file)
        objects = label['frames'][0]['objects']

        if len(self.target_objects) > 0:
            objects = get_target_objects(objects, self.target_objects)
            if len(objects) == 0:
                return False

        if 'attributes' in label and self.with_attr:
            attributes = label['attributes']
            self.ax.text(
                25 * self.scale, 90 * self.scale,
                '  scene: {}\nweather: {}\n   time: {}'.format(
                    attributes['scene'], attributes['weather'],
                    attributes['timeofday']),
                fontproperties=self.font,
                color='red',
                bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 10, 'lw': 0})

        if self.with_drivable:
            self.draw_drivable(objects)
        if self.with_lane:
            self.draw_lanes(objects)
        if self.with_box2d:
            [self.ax.add_patch(self.box2rect(b['box2d']))
             for b in get_boxes(objects)]
        if self.with_segment:
            self.draw_segments(objects)
        self.ax.axis('off')
        return True

    def next_image(self, event):
        if event.key == 'n':
            self.current_index += 1
        elif event.key == 'p':
            self.current_index -= 1
        else:
            return
        self.current_index = max(min(self.current_index,
                                     len(self.image_paths) - 1), 0)
        if self.show_image():
            plt.draw()
        else:
            self.next_image(event)

    def poly2patch(self, poly2d, closed=False, alpha=1., color=None):
        moves = {'L': Path.LINETO,
                 'C': Path.CURVE4}
        points = [p[:2] for p in poly2d]
        codes = [moves[p[2]] for p in poly2d]
        codes[0] = Path.MOVETO

        if closed:
            points.append(points[0])
            if codes[-1] == 4:
                codes.append(Path.LINETO)
            else:
                codes.append(Path.CLOSEPOLY)

        if color is None:
            color = random_color()

        # print(codes, points)
        return mpatches.PathPatch(
            Path(points, codes),
            facecolor=color if closed else 'none',
            edgecolor=color,  # if not closed else 'none',
            lw=1 if closed else 2 * self.scale, alpha=alpha,
            antialiased=False, snap=True)

    def draw_drivable(self, objects):
        objects = get_areas_v0(objects)
        colors = np.array([[0, 0, 0, 255],
                           [217, 83, 79, 255],
                           [91, 192, 222, 255]]) / 255
        for obj in objects:
            if self.color_mode == 'random':
                if obj['category'] == 'area/drivable':
                    color = colors[1]
                else:
                    color = colors[2]
                alpha = 0.5
            else:
                color = (
                    (1 if obj['category'] == 'area/drivable' else 2) / 255.,
                    obj['id'] / 255., 0)
                alpha = 1
            self.ax.add_patch(self.poly2patch(
                obj['poly2d'], closed=True, alpha=alpha, color=color))

    def draw_lanes(self, objects):
        objects = get_lanes(objects)
        # colors = np.array([[0, 0, 0, 255],
        #                    [217, 83, 79, 255],
        #                    [91, 192, 222, 255]]) / 255
        colors = np.array([[0, 0, 0, 255],
                           [255, 0, 0, 255],
                           [0, 0, 255, 255]]) / 255
        for obj in objects:
            if self.color_mode == 'random':
                if obj['attributes']['direction'] == 'parallel':
                    color = colors[1]
                else:
                    color = colors[2]
                alpha = 0.9
            else:
                color = (
                    (1 if obj['category'] == 'area/drivable' else 2) / 255.,
                    obj['id'] / 255., 0)
                alpha = 1
            self.ax.add_patch(self.poly2patch(
                obj['poly2d'], closed=False, alpha=alpha, color=color))

    def draw_segments(self, objects):
        color_mode = self.color_mode
        for obj in objects:
            if 'segments2d' not in obj:
                continue
            if color_mode == 'random':
                color = random_color()
                alpha = 0.5
            elif color_mode == 'instance':
                try:
                    label = self.label_map[obj['category']]
                    color = (label.trainId / 255., obj['id'] / 255., 0)
                except KeyError:
                    color = (1, 0, 0)
                alpha = 1
            else:
                raise ValueError('Unknown color mode {}'.format(
                    self.color_mode))
            for segment in obj['segments2d']:
                self.ax.add_patch(self.poly2patch(
                    segment, closed=True, alpha=alpha, color=color))

    def box2rect(self, box2d):
        """generate individual bounding box from label"""
        x1 = box2d['x1']
        y1 = box2d['y1']
        x2 = box2d['x2']
        y2 = box2d['y2']

        # Pick random color for each box
        box_color = random_color()

        # Draw and add one box to the figure
        return mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3 * self.scale, edgecolor=box_color, facecolor='none',
            fill=False, alpha=0.75
        )


def read_labels(label_path):
    labels = json.load(open(label_path, 'r'))
    if not isinstance(labels, Iterable):
        labels = [labels]
    return labels


class LabelViewer2(object):
    def __init__(self, args):
        """Visualize bounding boxes"""
        self.ax = None
        self.fig = None
        self.frame_index = 0
        self.file_index = 0
        self.label = None
        self.start_index = 0
        self.scale = args.scale
        if isdir(args.label):
            input_names = sorted(
                [splitext(n)[0] for n in os.listdir(args.label)
                 if splitext(n)[1] == '.json'])
            label_paths = [join(args.label, n + '.json') for n in input_names]
        else:
            label_paths = [args.label]
        self.label_paths = label_paths
        self.image_dir = args.image_dir

        self.font = FontProperties()
        self.font.set_family(['Luxi Mono', 'monospace'])
        self.font.set_weight('bold')
        self.font.set_size(18 * self.scale)

        self.with_image = True
        self.with_attr = not args.no_attr
        self.with_lane = not args.no_lane
        self.with_drivable = not args.no_drivable
        self.with_box2d = not args.no_box2d
        self.poly2d = True

        self.target_objects = args.target_objects

        if len(self.target_objects) > 0:
            print('Only showing objects:', self.target_objects)

        self.out_dir = args.output_dir
        self.label_map = dict([(l.name, l) for l in labels])
        self.color_mode = 'random'
        self.label_colors = {}

        self.image_width = 1280
        self.image_height = 720

        self.instance_mode = False
        self.drivable_mode = False
        self.with_post = False  # with post processing

        if args.drivable:
            self.set_drivable_mode()

        if args.instance:
            self.set_instance_mode()

        self.label = read_labels(self.label_paths[self.file_index])

    def view(self):
        self.frame_index = 0
        if self.out_dir is None:
            self.show()
        else:
            self.write()

    def show(self):
        # Read and draw image
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        # if len(self.image_paths) > 1:
        plt.connect('key_release_event', self.next_image)
        self.show_image()
        plt.show()

    def write(self):
        dpi = 80
        w = 16
        h = 9
        self.fig = plt.figure(figsize=(w, h), dpi=dpi)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

        out_paths = []

        self.start_index = 0
        self.frame_index = 0
        self.file_index = 0
        while self.file_index < len(self.label_paths):
            if self.label is None:
                self.label = read_labels(self.label_paths[self.file_index])
            out_name = splitext(split(
                self.label[self.frame_index -
                           self.start_index]['name'])[1])[0] + '.png'
            out_path = join(self.out_dir, out_name)
            if self.show_image():
                self.fig.savefig(out_path, dpi=dpi)
                out_paths.append(out_path)
            self.frame_index += 1
            if self.frame_index >= len(self.label):
                self.start_index = self.frame_index
                self.file_index += 1
                self.label = None

        if self.with_post:
            print('Post-processing')
            p = Pool(10)
            if self.instance_mode:
                p.map(convert_instance_rgb, out_paths)
            if self.drivable_mode:
                p = Pool(10)
                p.map(convert_drivable_rgb, out_paths)

    def set_instance_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = False
        self.with_lane = False
        self.with_box2d = False
        self.poly2d = True
        self.color_mode = 'instance'
        self.instance_mode = True
        self.with_post = True

    def set_drivable_mode(self):
        self.with_image = False
        self.with_attr = False
        self.with_drivable = True
        self.with_lane = False
        self.with_box2d = False
        self.poly2d = False
        self.color_mode = 'instance'
        self.drivable_mode = True
        self.with_post = True

    def show_image(self):
        plt.cla()
        if self.frame_index >= self.start_index + len(self.label):
            self.label = None
            self.file_index += 1
            self.start_index = self.frame_index
            if self.file_index >= len(self.label_paths):
                self.file_index = 0
                self.frame_index = 0
                self.start_index = 0
        if self.label is None:
            self.label = read_labels(self.label_paths[self.file_index])

        frame = self.label[self.frame_index - self.start_index]

        print('Image:', frame['name'])
        self.fig.canvas.set_window_title(frame['name'])

        if self.with_image:
            if 'url' in frame and len(frame['url']) > 0:
                req = urllib.request.Request(frame['url'])
                image_data = urllib.request.urlopen(req, timeout=300).read()
                im = np.asarray(Image.open(io.BytesIO(image_data)))
            else:
                image_path = join(self.image_dir, frame['name'])
                img = mpimg.imread(image_path)
                im = np.array(img, dtype=np.uint8)
            self.ax.imshow(im, interpolation='nearest', aspect='auto')
        else:
            self.ax.set_xlim(0, self.image_width - 1)
            self.ax.set_ylim(0, self.image_height - 1)
            self.ax.invert_yaxis()
            self.ax.add_patch(self.poly2patch(
                [[0, 0], [0, self.image_height - 1],
                 [self.image_width - 1, self.image_height - 1],
                 [self.image_width - 1, 0]], types='LLLL',
                closed=True, alpha=1., color='black'))

        if 'labels' not in frame or frame['labels'] is None:
            print('No labels')
            return True

        objects = frame['labels']

        calibration = None
        if 'intrinsics' in frame and 'cali' in frame['intrinsics']:
            calibration = np.array(frame['intrinsics']['cali'])

        if len(self.target_objects) > 0:
            objects = get_target_objects(objects, self.target_objects)
            if len(objects) == 0:
                return False

        if self.with_attr:
            self.show_attributes(frame)

        if self.with_drivable:
            self.draw_drivable(objects)
        if self.with_lane:
            self.draw_lanes(objects)
        if self.with_box2d:
            for b in get_boxes(objects):
                if 'box3d' in b:
                    occluded = False
                    if 'Occluded' in b['attributes']:
                        occluded = b['attributes']['Occluded']

                    for line in self.box3d_to_lines(
                            b['id'], b['box3d'], calibration, occluded):
                        self.ax.add_patch(line)
                else:
                    self.ax.add_patch(self.box2rect(b['id'], b['box2d']))
        if self.poly2d:
            self.draw_other_poly2d(objects)
        self.ax.axis('off')
        return True

    def next_image(self, event):
        if event.key == 'n':
            self.frame_index += 1
        elif event.key == 'p':
            self.frame_index -= 1
        else:
            return
        self.frame_index = max(self.frame_index, 0)
        if self.show_image():
            plt.draw()
        else:
            self.next_image(event)

    def poly2patch(self, vertices, types, closed=False, alpha=1., color=None):
        moves = {'L': Path.LINETO,
                 'C': Path.CURVE4}
        points = [v for v in vertices]
        codes = [moves[t] for t in types]
        codes[0] = Path.MOVETO

        if closed:
            points.append(points[0])
            codes.append(Path.CLOSEPOLY)

        if color is None:
            color = random_color()

        # print(codes, points)
        return mpatches.PathPatch(
            Path(points, codes),
            facecolor=color if closed else 'none',
            edgecolor=color,  # if not closed else 'none',
            lw=1 if closed else 2 * self.scale, alpha=alpha,
            antialiased=False, snap=True)

    def draw_drivable(self, objects):
        objects = get_areas(objects)
        colors = np.array([[0, 0, 0, 255],
                           [217, 83, 79, 255],
                           [91, 192, 222, 255]]) / 255
        for obj in objects:
            if self.color_mode == 'random':
                if obj['attributes']['areaType'] == 'direct':
                    color = colors[1]
                else:
                    color = colors[2]
                alpha = 0.5
            else:
                color = ((1 if obj['attributes']['areaType'] ==
                         'direct' else 2) / 255.,
                         (obj['id'] // 255) / 255,
                         (obj['id'] % 255) / 255.)
                alpha = 1
            for poly in obj['poly2d']:
                self.ax.add_patch(self.poly2patch(
                    poly['vertices'], poly['types'], closed=poly['closed'],
                    alpha=alpha, color=color))

    def draw_lanes(self, objects):
        objects = get_lanes(objects)
        # colors = np.array([[0, 0, 0, 255],
        #                    [217, 83, 79, 255],
        #                    [91, 192, 222, 255]]) / 255
        colors = np.array([[0, 0, 0, 255],
                           [255, 0, 0, 255],
                           [0, 0, 255, 255]]) / 255
        for obj in objects:
            if self.color_mode == 'random':
                if obj['attributes']['laneDirection'] == 'parallel':
                    color = colors[1]
                else:
                    color = colors[2]
                alpha = 0.9
            else:
                color = (0, (obj['id'] // 255) / 255, (obj['id'] % 255) / 255.)
                alpha = 1
            for poly in obj['poly2d']:
                self.ax.add_patch(self.poly2patch(
                    poly['vertices'], poly['types'], closed=poly['closed'],
                    alpha=alpha, color=color))

    def draw_other_poly2d(self, objects):
        color_mode = self.color_mode
        objects = get_other_poly2d(objects)
        for obj in objects:
            if 'poly2d' not in obj:
                continue
            if color_mode == 'random':
                color = self.get_label_color(obj['id'])
                alpha = 0.5
            elif color_mode == 'instance':
                try:
                    label = self.label_map[obj['category']]
                    color = (label.trainId / 255., (obj['id'] // 255) / 255,
                             (obj['id'] % 255) / 255.)
                except KeyError:
                    color = (1, 0, 0)
                alpha = 1
            else:
                raise ValueError('Unknown color mode {}'.format(
                    self.color_mode))
            for poly in obj['poly2d']:
                self.ax.add_patch(self.poly2patch(
                    poly['vertices'], poly['types'], closed=poly['closed'],
                    alpha=alpha, color=color))

    def box2rect(self, label_id, box2d):
        """generate individual bounding box from label"""
        x1 = box2d['x1']
        y1 = box2d['y1']
        x2 = box2d['x2']
        y2 = box2d['y2']

        box_color = self.get_label_color(label_id)

        # Draw and add one box to the figure
        return mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3 * self.scale, edgecolor=box_color, facecolor='none',
            fill=False, alpha=0.75
        )

    def box3d_to_lines(self, label_id, box3d, calibration, occluded):
        """generate individual bounding box from 3d label"""
        label = Label3d.from_box3d(box3d)
        edges = label.get_edges_with_visibility(calibration)

        box_color = self.get_label_color(label_id)
        alpha = 0.5 if occluded else 0.8

        lines = []
        for edge in edges['dashed']:
            lines.append(mpatches.Polygon(edge, linewidth=2 * self.scale,
                                          linestyle=(0, (2, 2)),
                                          edgecolor=box_color,
                                          facecolor='none', fill=False,
                                          alpha=alpha))
        for edge in edges['solid']:
            lines.append(mpatches.Polygon(edge, linewidth=2 * self.scale,
                                          edgecolor=box_color,
                                          facecolor='none', fill=False,
                                          alpha=alpha))

        return lines

    def get_label_color(self, label_id):
        if label_id not in self.label_colors:
            self.label_colors[label_id] = random_color()
        return self.label_colors[label_id]

    def show_attributes(self, frame):
        if 'attributes' not in frame:
            return
        attributes = frame['attributes']
        if attributes is None or len(attributes) == 0:
            return
        key_width = 0
        for k, _ in attributes.items():
            if len(k) > key_width:
                key_width = len(k)
        attr_tag = io.StringIO()
        for k, v in attributes.items():
            attr_tag.write('{}: {}\n'.format(
                k.rjust(key_width, ' '), v))
        attr_tag.seek(0)
        self.ax.text(
            25 * self.scale, 90 * self.scale, attr_tag.read()[:-1],
            fontproperties=self.font,
            color='red',
            bbox={'facecolor': 'white', 'alpha': 0.4, 'pad': 10, 'lw': 0})


def main():
    args = parse_args()
    if args.format == 'v1':
        viewer = LabelViewer(args)
    else:
        viewer = LabelViewer2(args)
    viewer.view()


if __name__ == '__main__':
    main()
