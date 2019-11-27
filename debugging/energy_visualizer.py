"""Energy Visualizer.

Just to make sure that the energy targets are correctly calculated.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import numpy as np
from PIL import Image

from math import floor, ceil

from .coco_dataset import CocoDetection
from .colormapping import map_color_values
from .bbox import BoundingBox
from . import draw_boxes

from os.path import join


def energy_target(flattened_bbox_targets, pos_bbox_targets,
                  pos_indices, r, max_energy):
    """Calculate energy targets based on deep watershed paper.

    Args:
        flattened_bbox_targets (torch.Tensor): The flattened bbox targets.
        pos_bbox_targets (torch.Tensor): Bounding box lrtb values only for
            positions within the bounding box. We use this as an argument
            to prevent recalculating it since it is used for other things as
            well.
        pos_indices (torch.Tensor): The indices of values in
            flattened_bbox_targets which are within a bounding box
        max_energy (int): Max energy level possible.

    Notes:
        The energy targets are calculated as:
        E_max \cdot argmax_{c \in C}[1 - \sqrt{((l-r)/2)^2 + ((t-b) / 2)^2}
                                     / r]

        - r is a hyperparameter we would like to minimize.
        - (l-r)/2 is the horizontal distance to the center and will be
            assigned the variable name "horizontal"
        - (t-b)/2 is the vertical distance to the center and will be
            assigned the variable name "vertical"
        - E_max is self.max_energy
        - We don't need the argmax in this code implementation since we
            already select the bounding boxes and their respective pixels in
            a previous step.

    Returns:
        tuple: A 2 tuple with values ("pos_energies_targets",
            "energies_targets"). Both are flattened but pos_energies_targets
            only contains values within bounding boxes.
    """

    horizontal = pos_bbox_targets[:, 0] - pos_bbox_targets[:, 2]
    vertical = pos_bbox_targets[:, 1] - pos_bbox_targets[:, 3]

    # print("Horizontals: {}".format(horizontal))
    # print("Verticals: {}".format(vertical))

    horizontal = torch.div(horizontal, 2)
    vertical = torch.div(vertical, 2)

    c2 = (horizontal * horizontal) + (vertical * vertical)

    # print("c2: \n{}".format(c2))

    # We use x * x instead of x.pow(2) since it's faster by about 30%
    square_root = torch.sqrt(c2)

    # print("Sqrt: \n{}".format(square_root))

    type_dict = {'dtype': square_root.dtype,
                 'device': square_root.device}

    pos_energies = (torch.tensor([1], **type_dict)
                    - torch.div(square_root, r))
    pos_energies *= max_energy
    pos_energies = torch.max(pos_energies,
                             torch.tensor([0], **type_dict))
    pos_energies = pos_energies.floor()

    energies_targets = torch.zeros(flattened_bbox_targets.shape[0],
                                   **type_dict)
    energies_targets[pos_indices] = pos_energies

    # torch.set_printoptions(profile='full')
    # print("Energy targets: \n {}".format(pos_energies))
    # torch.set_printoptions(profile='default')
    # input()

    return pos_energies, energies_targets


def process_target(img, targets):
    """Processes targets and returns an overlaid image."""
    bboxes = BoundingBox()
    for t in targets:
        bboxes.append_target(t['bbox'], t['category_id'])
    return draw_boxes(img, bboxes, threshold=1., target=True)


def overlay_energy(image, energies_targets) -> Image.Image:
    energy_overlay = map_color_values(energies_targets.numpy(), 20)
    energy_overlay = Image.fromarray(energy_overlay)

    return Image.blend(image, energy_overlay, 0.4)


def visualize_image(image, annotations, output_dir, iterator):
    """Visualizes the energy of a single image and outputs it to a file."""
    # Get bboxes first
    bboxes = [annotation['bbox'] for annotation in annotations]

    # These are both not flattened yet
    pos_bbox_targets = []
    pos_indices = []

    width, height = image.size

    # For each pixel within the bounding boxes, calculate the lrtb box values
    # of that pixel
    # We do this by iterating through each pixel of the image
    for i in range(height):
        for j in range(width):
            for bbox in bboxes:
                left, top, bbox_width, bbox_height = bbox
                right = left + bbox_width
                bottom = top + bbox_height
                if ceil(left) <= j <= floor(right) \
                        and ceil(top) <= i <= floor(bottom):
                    # If the pixel is within the bounding box
                    dist_left = float(j) - left
                    dist_right = right - float(j)
                    dist_top = float(i) - top
                    dist_bottom = bottom - float(i)

                    pos_bbox_targets.append([dist_left, dist_top,
                                             dist_right, dist_bottom])
                    pos_indices.append((width * i) + j)

    pos_bbox_targets = np.array(pos_bbox_targets)
    flattened_bboxes = np.zeros([width * height, 4])
    flattened_bboxes[pos_indices] = pos_bbox_targets

    pos_bbox_targets = torch.tensor(pos_bbox_targets)
    flattened_bboxes = torch.tensor(flattened_bboxes)
    pos_indices = torch.tensor(pos_indices)

    pos_energies, energies_targets = energy_target(flattened_bboxes,
                                                   pos_bbox_targets,
                                                   pos_indices,
                                                   300.,
                                                   20)

    energies_targets = energies_targets.reshape([height, width])

    overlaid = overlay_energy(image, energies_targets)

    overlaid = process_target(overlaid, annotations)

    overlaid.save(join(output_dir, str(iterator) + '.png'))


def runner(coco_dataset_root, output_fp):
    """Runs the program."""
    ann_fp = join(coco_dataset_root, 'annotations', 'instances_small2017.json')
    dataset = CocoDetection(join(coco_dataset_root, 'images', 'val2017'),
                            ann_fp)

    stop = False
    i = 0

    while not stop:
        img, annotation = dataset[i]
        if len(annotation) != 0:
            visualize_image(img, annotation, output_fp, i)
            response = input('continue? [yes]')
            if response == 'no' or response == 'n':
                stop = True
        i += 1


if __name__ == '__main__':
    runner('/workspace/coco/', '/workspace/watershed-fcos/work_dirs/energy_vis')
