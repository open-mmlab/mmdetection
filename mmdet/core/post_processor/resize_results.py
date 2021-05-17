from mmdet.core import bbox_flip
from mmdet.core.post_processor.builder import POST_PROCESSOR


@POST_PROCESSOR.register_module()
class ResultsToOri(object):
    """Mapping results to the scale of original image.

    Args:
        results_types (str | list[str]): Used to specify the
            result type to resize.
    """

    def __init__(self, results_types=['bbox']):
        support_types = ['bbox', 'masks']
        if isinstance(results_types, str):
            results_types = [results_types]
        for r_type in results_types:
            if r_type not in support_types:
                raise NotImplementedError(
                    f'ResultsToOri'
                    f' only support resize{support_types}'
                    f' to original image, '
                    f' but get {results_types}')
        self.results_type = results_types

    def __call__(self, results_list):
        # shape(n, num_class), the num_class does not
        # include the background
        r_results_list = []
        for results in results_list:
            for _type in self.results_type:
                results = getattr(self, f'_mapping_{_type}')(results)
            r_results_list.append(results)

        return r_results_list

    def _mapping_bbox(self, results):
        scale_factor = results.bboxes.new_tensor(results.scale_factor)
        if results.flip:
            results.bboxes = bbox_flip(results.bboxes, results.img_shape)
        results.bboxes = results.bboxes / scale_factor
        return results
