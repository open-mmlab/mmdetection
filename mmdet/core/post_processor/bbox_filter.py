from mmdet.core.post_processor.builder import POST_PROCESSOR


@POST_PROCESSOR.register_module()
class BoxFilter(object):
    """Remove the bboxes smaller than the min_bbox_size.

    Args:
        min_bbox_size (float): bbox_size threshold.
            Defaults to 0.
    """

    def __init__(self, min_bbox_size=0.):
        self.min_bbox_size = min_bbox_size

    def __call__(self, results_list):
        processed_results_list = []
        for results in results_list:
            bboxes = results.bboxes
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            valid_mask = (w > self.min_bbox_size) & (h > self.min_bbox_size)

            if valid_mask.sum() == len(results):
                processed_results = results[valid_mask]
            else:
                processed_results = results
            processed_results_list.append(processed_results)

        return processed_results_list
