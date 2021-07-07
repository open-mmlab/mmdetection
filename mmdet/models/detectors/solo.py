from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class SOLO(SingleStageInstanceSegmentor):

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        raise NotImplementedError
