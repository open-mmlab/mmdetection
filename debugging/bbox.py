"""Bounding Box.

Bounding box class that represents abstract bounding boxes.
"""
import torch

from classes_lookup import *
from constants import *


class BoundingBox:
    def __init__(self):
        """Creates a list of bounding boxes from the network output.

        Creates an abstract bounding box class from the network output. This
        class keeps track of the individual bounding boxes, and adds class
        labels to them based on the segmentation output from the network.

        One BoundingBox object should be created per batch.

        Usage:
            1.  Initialize BoundingBox.
            2.  Append tensors to BoundingBox using BoundingBox.append()
            3a. To get suppressed bounding boxes, use
                BoundingBox.get_suppressed()
            3b. To get suppressed bounding boxes using watershed, use
                BoundingBox.get_watershed()
        """
        # BBoxes are now stored in (n, 6) form with 6 being (l, r, t, b, cat,
        # score)
        self.bboxes = torch.empty([0, 6]).to(device=DEVICE)

    def append_target(self, box, category):
        """Apends target bbox to the list of bounding boxes.

        Args:
            bbox (list): A list of length 4 whose contents are tensors. It has
                the values [l, t, width, height]
            category (torch.Tensor): Category number of the bounding box.
        """
        # Hacky way to get proper category name
        # Iterate through cats list to find id
        bbox = [box[0]] + [box[1]] + [box[0] + box[2]] + [box[1] + box[3]] \
               + [category] + [torch.tensor(5.)]

        bbox = torch.tensor(bbox, dtype=torch.float).unsqueeze(0)\
            .to(device=DEVICE)
        self.bboxes = torch.cat([self.bboxes, bbox])


    def append(self, bbox_preds, cls_preds, score_preds, m):
        """Appends a head output to the list of bounding boxes.

        Shape:
            bbox_output: (4, h, w) with the first 4 being (left, top, right,
                bottom) distance from the bounding box edge.

            cls_output: (80, h, w) with the first 80 being the classes. This
                does not have to be softmaxed.

            centerness_output: (1, h, w) with the first dimension having a range
                in the interval (0, 1)

            m: (2) scaling factor on the x, y axis, respectively.

        Args:
            bbox_preds (torch.Tensor): The bbox output of the network.
            cls_preds (torch.Tensor): The class segmentation output of the
                network.
            score_preds (torch.Tensor): The centerness output of the
                network.
            m (torch.Tensor): The scaling factor to scale the head output back
                to the original image size.
        """
        assert bbox_preds.shape[0] == 4
        assert cls_preds.shape[0] == 80
        assert score_preds.shape[0] == 1
        assert m.shape[0] == 2

        # get argmax from score_preds
        score_preds = score_preds.argmax(0).reshape(1, score_preds.shape[1],
                                                    score_preds.shape[2])


        # BBoxes are stored as a (n, 6) tensor, with n being the number of
        # bboxes and the 6 being (x0, y0, x1, y1, cat, score)
        cls_preds = cls_preds.softmax(0)
        cls_preds = cls_preds.argmax(0)
        bbox_preds = self._resize_bbox(bbox_preds, m)
        bbox_preds = self._append_class_score(bbox_preds, cls_preds,
                                              score_preds)
        self.bboxes = torch.cat((self.bboxes, bbox_preds))

    def _resize_bbox(self, preds, m):
        """Turns bboxes into the standard (l,t,r,b) format and resizes them.

        Gets the bounding boxes in the standard format. Also suppresses any
        bounding boxes that are non-max. This is done simply by not doing
        operations on grid values where max_values is not 1.

        Shape:
            preds: (4, h, w) as returned by the individual head of the network.

            m: (2) scaling factor on the x, y axis respectively.

            max_values: (1, h, w) as returned by bbox_select()

        Args:
            preds (torch.Tensor): A tensor that is returned by the network and
                represents the bounding boxes. This tensor should be in
                (l, t, r, b) format, where l, t, r, and b are the left, top,
                right, and bottom distance to the edge of the bounding box.
            m (torch.Tensor): The scaling factor to scale the head output back
                to the image size.
            max_values (torch.Tensor): A byte tensor where bboxes that should be
                processed are 1 and those that should not be are 0.

        Notes:

        Bounding boxes are in the form (x0, y0, x1, y1), where x0 is the x value
        of the left edge, y0 the y value of the top edge, x1 the x value of the
        right edge, and y1 the y value of the bottom edge. This is a standard
        representation for bounding boxes.

        The network returns bounding box values for each pixel with coordinates
        (x, y) in the form (l, t, r, b), with l = x - x0, t = y - y0,
        r = x1 - x, b = y1 - y.

        To get the bounding boxes in standard form so we can display it, we
        require the (x0, y0, x1, y1) values from each pixel with:
        x0 = -l + (m * x) + s_x
        y0 = -t + (m * y) + s_y
        x1 = r + (m * x) + s_x
        y1 = b + (m * y) + s_y

        with m being the scaling factor to get the size of the head back to the
        actual size of the image and s being the shift value to move the values
        into the correct position.

        A shift value s is needed since we draw with the origin located in the
        top left, whereas the distance values are computed from the center of a
        pixel.

        The variable ONES is the required multiplier to turn (l, t, r, b) into
        (-l, -t, r, b). The variable INDEX_TENSOR contains the index values of
        each pixel, i.e. (x, y, x, y).

        Returns:
            torch.Tensor: The standardized bboxes in the shape (4, h, w).
        """
        m = m.reshape(2, 1).repeat(2, 1)

        preds *= ONES[:, :preds.shape[1], :preds.shape[2]]
        # Get m * index tensor
        a = INDEX_TENSOR[:, :preds.shape[1], :preds.shape[2]].clone()
        m_xy = a.transpose(0, 1)
        m_xy *= m
        m_xy = m_xy.transpose(0, 1)

        preds += m_xy

        # Get shift s_xy
        b = HALVES[:, :preds.shape[1], :preds.shape[2]].clone()
        s_xy = b.transpose(0, 1)
        s_xy *= m
        s_xy = s_xy.transpose(0, 1)

        preds += s_xy

        return preds

    def get_suppressed(self, threshold):
        """Suppresses bbox values below a certain score.

        Shape:
            bbox_preds: (h, w, 5), the 5 being (x0, y0, x1, y1, cls)

            centerness: (1, h, w)

        Args:
            bbox_preds (torch.Tensor): Bounding Box predictions from the
                network.
            centerness (torch.Tensor): Centerness values from the network.
            threshold (float): The threshold to display a bounding box.

        Returns:
            torch.Tensor: Torch tensor in the shape (n, 5) containing only
                bboxes with centerness values above the threshold where n is the
                number of bboxes.
        """
        return self.bboxes[self.bboxes[:, 5].gt(threshold)]

    def get_watershed(self, percentage):
        """Suppresses bboxes with scores that are below the percentage value.

        Args:
            percentage: Percentage to suppress. For example, if percentage is
                .80 and there are 100 bboxes total, 20 bboxes will be returned.
        """
        num_bboxes = int(self.bboxes.shape[0] * percentage)
        scores = self.bboxes[:, 5]
        kth_value = scores.kthvalue(num_bboxes)

        return self.bboxes[scores.gt(kth_value.values)]

    def _append_class_score(self, bbox_preds, cls_preds, score_preds):
        """Appends class and scores, and permutes to shape (n, 6).

        Shape:
            bbox_preds: (4, h, w)
            cls_preds: (h, w)
            score_preds: (1, h, w)

        Args:
            bbox_preds (torch.Tensor): BBox predictions.
            cls_preds (torch.Tensor): Class predictions.
            score_preds (torch.Tensor): Score predictions, either energy or
                centerness.

        Returns:
            torch.Tensor: (n, 6) tensor
        """
        cls_preds = cls_preds.unsqueeze(-1)
        score_preds = score_preds.permute(1, 2, 0)
        score_preds += abs(score_preds.min())
        normalizer = score_preds.max()
        # # Normalizing values are handselected.
        # if score_preds.shape[0] >= 20:
        #     normalizer = score_preds.max()
        # elif score_preds.shape[0] == 10:
        #     normalizer = 3.
        # elif score_preds.shape[0] == 5:
        #     normalizer = score_preds.max()
        # else:
        #     raise NotImplementedError

        # TODO Something breaks here
        if normalizer == 0:
            normalizer = 1

        score_preds /= abs(normalizer)
        bbox_preds = bbox_preds.permute(1, 2, 0)

        out = torch.cat((bbox_preds, cls_preds.to(bbox_preds.dtype)), 2)
        out = torch.cat((out, score_preds.to(out.dtype)), 2)

        return out.flatten(start_dim=0, end_dim=1)
