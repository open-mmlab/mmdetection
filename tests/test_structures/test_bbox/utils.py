from mmdet.structures.bbox import BaseBoxes


class ToyBaseBoxes(BaseBoxes):

    box_dim = 4

    @property
    def centers(self):
        pass

    @property
    def areas(self):
        pass

    @property
    def widths(self):
        pass

    @property
    def heights(self):
        pass

    def flip_(self, img_shape, direction='horizontal'):
        pass

    def translate_(self, distances):
        pass

    def clip_(self, img_shape):
        pass

    def rotate_(self, center, angle):
        pass

    def project_(self, homography_matrix):
        pass

    def rescale_(self, scale_factor):
        pass

    def resize_(self, scale_factor):
        pass

    def is_inside(self, img_shape):
        pass

    def find_inside_points(self, points, is_aligned=False):
        pass

    def overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        pass

    def from_instance_masks(masks):
        pass
