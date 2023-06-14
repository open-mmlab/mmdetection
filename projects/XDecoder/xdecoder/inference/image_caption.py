import copy
import os.path as osp
from typing import Iterable, List, Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.dataset import Compose
from rich.progress import track

from mmdet.apis.det_inferencer import DetInferencer, InputsType, PredType
from mmdet.utils import ConfigType


def get_adaptive_scale(img_shape: Tuple[int, int],
                       min_scale: float = 0.3,
                       max_scale: float = 3.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_scale (float): The minimum scale. Defaults to 0.3.
        max_scale (float): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.
    return min(max(scale, min_scale), max_scale)


class ImageCaptionInferencer(DetInferencer):
    DEFAULT_TEXT_CFG = {
        'font_families': 'monospace',
        'colors': 'white',
        'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
        'vertical_alignments': 'top',
        'horizontal_alignments': 'left',
    }

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  **kwargs) -> Union[List[np.ndarray], None]:

        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if self.visualizer is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        text_cfg = self.DEFAULT_TEXT_CFG

        for single_input, pred in zip(inputs, preds):
            if isinstance(single_input, str):
                img_bytes = mmengine.fileio.get(single_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(single_input)
            elif isinstance(single_input, np.ndarray):
                img = single_input.copy()
                img_num = str(self.num_visualized_imgs).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(single_input)}')

            out_file = osp.join(img_out_dir, 'vis',
                                img_name) if img_out_dir != '' else None

            self.visualizer.set_image(img)

            img_scale = get_adaptive_scale(img.shape[:2])
            text_cfg['font_sizes'] = int(img_scale * 7)

            self.visualizer.draw_texts(
                pred.pred_caption, torch.tensor([img_scale * 5,
                                                 img_scale * 5]), **text_cfg)
            drawn_img = self.visualizer.get_image()

            self.visualizer.add_datasample(
                img_name,
                drawn_img,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                out_file=out_file,
            )
            results.append(self.visualizer.get_image())
            self.num_visualized_imgs += 1

        return results


class RefImageCaptionInferencer(ImageCaptionInferencer):

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        # For inference, the key of ``img_id`` is not used.
        if 'meta_keys' in pipeline_cfg[-1]:
            pipeline_cfg[-1]['meta_keys'] = tuple(
                meta_key for meta_key in pipeline_cfg[-1]['meta_keys']
                if meta_key != 'img_id')

        load_img_idx = self._get_transform_idx(pipeline_cfg,
                                               'LoadImageFromFile')
        if load_img_idx == -1:
            raise ValueError(
                'LoadImageFromFile is not found in the test pipeline')
        pipeline_cfg[load_img_idx]['type'] = 'mmdet.InferencerLoader'

        caption_pipeline = Compose(pipeline_cfg)

        grounding_pipeline_cp = copy.deepcopy(pipeline_cfg)
        grounding_pipeline_cp[1].scale = cfg.grounding_scale
        grounding_pipeline = Compose(grounding_pipeline_cp)

        return {
            'grounding_pipeline': grounding_pipeline,
            'caption_pipeline': caption_pipeline
        }

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    if 'img' in inputs_:
                        ori_inputs_ = inputs_['img']
                    else:
                        ori_inputs_ = inputs_['img_path']
                    chunk_data.append(
                        (ori_inputs_, self.pipeline['grounding_pipeline'](
                            copy.deepcopy(inputs_)),
                         self.pipeline['caption_pipeline'](
                             copy.deepcopy(inputs_))))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def __call__(
            self,
            inputs: InputsType,
            batch_size: int = 1,
            return_vis: bool = False,
            show: bool = False,
            wait_time: int = 0,
            no_save_vis: bool = False,
            draw_pred: bool = True,
            pred_score_thr: float = 0.3,
            return_datasample: bool = False,
            print_result: bool = False,
            no_save_pred: bool = True,
            out_dir: str = '',
            texts: Optional[Union[str, list]] = None,
            # by open panoptic task
            stuff_texts: Optional[Union[str, list]] = None,
            custom_entities: bool = False,  # by GLIP
            **kwargs) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            batch_size (int): Inference batch size. Defaults to 1.
            show (bool): Whether to display the visualization results in a
                popup window. Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            no_save_vis (bool): Whether to force not to save prediction
                vis results. Defaults to False.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            return_datasample (bool): Whether to return results as
                :obj:`DetDataSample`. Defaults to False.
            print_result (bool): Whether to print the inference result w/o
                visualization to the console. Defaults to False.
            no_save_pred (bool): Whether to force not to save prediction
                results. Defaults to True.
            out_file: Dir to save the inference results or
                visualization. If left as empty, no file will be saved.
                Defaults to ''.

            **kwargs: Other keyword arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``, ``visualize_kwargs``
                and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """
        assert batch_size == 1
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        ori_inputs = self._inputs_to_list(inputs)

        if isinstance(texts, str):
            texts = [texts] * len(ori_inputs)

        for i in range(len(texts)):
            if isinstance(ori_inputs[i], str):
                ori_inputs[i] = {
                    'text': texts[i],
                    'img_path': ori_inputs[i],
                    'custom_entities': custom_entities
                }
            else:
                ori_inputs[i] = {
                    'text': texts[i],
                    'img': ori_inputs[i],
                    'custom_entities': custom_entities
                }
        inputs = self.preprocess(
            ori_inputs, batch_size=batch_size, **preprocess_kwargs)

        results_dict = {'predictions': [], 'visualization': []}
        for ori_inputs, grounding_data, caption_data in track(
                inputs, description='Inference'):

            self.model.sem_seg_head.task = 'ref-seg'
            self.model.sem_seg_head.predictor.task = 'ref-seg'
            preds = self.forward(grounding_data, **forward_kwargs)

            for data_sample, pred_datasmaple in zip(
                    caption_data['data_samples'], preds):
                data_sample.pred_instances = pred_datasmaple.pred_instances
                data_sample.set_metainfo({
                    'grounding_img_shape':
                    pred_datasmaple.metainfo['img_shape']
                })

            self.model.sem_seg_head.task = 'caption'
            self.model.sem_seg_head.predictor.task = 'caption'

            preds = self.forward(caption_data, **forward_kwargs)

            if isinstance(ori_inputs, dict):
                ori_inputs = ori_inputs['img_path']

            visualization = self.visualize(
                ori_inputs,
                preds,
                return_vis=return_vis,
                show=show,
                wait_time=wait_time,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                no_save_vis=no_save_vis,
                img_out_dir=out_dir,
                **visualize_kwargs)
            results = self.postprocess(
                preds,
                visualization,
                return_datasample=return_datasample,
                print_result=print_result,
                no_save_pred=no_save_pred,
                pred_out_dir=out_dir,
                **postprocess_kwargs)
            results_dict['predictions'].extend(results['predictions'])
            if results['visualization'] is not None:
                results_dict['visualization'].extend(results['visualization'])
        return results_dict
