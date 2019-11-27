from PIL import ImageDraw, ImageFont
from .classes_lookup import *
from .colormapping import map_color_value
from .constants import *

from os import getcwd


def draw_boxes(image, bounding_boxes, use_watershed=False, threshold=0.8,
               target=False):
    """Draw all bounding_boxes on the image along with their class.

    Args:
        image (PIL.Image.Image): Image to draw the bounding boxes on
        bounding_boxes (bbox.bbox.BoundingBox): List of BoundingBox objects.
            Each BoundingBox object contains all bounding boxes of an image.
        use_watershed (bool): Use watershed or a simple threshold value
        threshold (float): Threshold or percentage value for suppression.
        target (bool): If the image comes from a target source

    Returns:
        PIL.Image.Image: Image with bounding boxes drawn on top.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    if use_watershed:
        pruned_bboxes = bounding_boxes.get_watershed(threshold)
    else:
        pruned_bboxes = bounding_boxes.get_suppressed(threshold)
    pruned_bboxes = pruned_bboxes.detach().to(device='cpu').numpy()

    # pruned_bboxes = bounding_boxes.detach().numpy()

    for bb in pruned_bboxes:
        # Map category to color
        if True: #int(bb[4]):
            color = tuple(map_color_value(bb[4], 80))

            # Get category class label
            if not target:
                label = CLASSES[int(bb[4])]
            else:
                label = CAT_DICT[int(bb[4])]

            # Draw the category label
            text_size = draw.textsize(label, FONT)
            draw.rectangle([bb[0], bb[1],
                            bb[0] + 6 + text_size[0],
                            bb[1] + 6 + text_size[1]],
                           fill=(0, 0, 0))
            draw.text(bb[0:2] + 3, label, fill=(255, 255, 255), font=FONT)

            # Draw the bounding box
            draw.rectangle(bb[0:4], outline=color, width=2)

    return img
