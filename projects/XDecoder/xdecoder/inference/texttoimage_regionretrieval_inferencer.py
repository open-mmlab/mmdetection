import copy
from typing import Iterable, Optional, Union

import torch
from mmengine.dataset import Compose
from rich.progress import track

from mmdet.apis.det_inferencer import DetInferencer, InputsType
from mmdet.utils import ConfigType


class TextToImageRegionRetrievalInferencer(DetInferencer):

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

        retrieval_pipeline = Compose(pipeline_cfg)

        grounding_pipeline_cp = copy.deepcopy(pipeline_cfg)
        grounding_pipeline_cp[1].scale = cfg.grounding_scale
        grounding_pipeline = Compose(grounding_pipeline_cp)

        return {
            'grounding_pipeline': grounding_pipeline,
            'retrieval_pipeline': retrieval_pipeline
        }

    def _get_chunk_data(self, inputs: Iterable, pipeline, chunk_size: int):
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
                    chunk_data.append(
                        (inputs_, pipeline(copy.deepcopy(inputs_))))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def preprocess(self,
                   inputs: InputsType,
                   pipeline,
                   batch_size: int = 1,
                   **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, pipeline, batch_size)
        yield from map(self.collate_fn, chunked_data)

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
            ori_inputs[i] = {
                'img_path': ori_inputs[i],
                'text': texts[i],
                'custom_entities': False
            }
        inputs = self.preprocess(
            ori_inputs,
            pipeline=self.pipeline['retrieval_pipeline'],
            batch_size=batch_size,
            **preprocess_kwargs)

        self.model.sem_seg_head._force_not_use_cache = True

        pred_scores = []
        for _, retrieval_data in track(inputs, description='Inference'):
            preds = self.forward(retrieval_data, **forward_kwargs)
            pred_scores.append(preds[0].pred_score)

        pred_score = torch.cat(pred_scores)
        pred_score = torch.softmax(pred_score, dim=0)
        max_id = torch.argmax(pred_score)
        retrieval_ori_input = ori_inputs[max_id.item()]
        max_prob = round(pred_score[max_id].item(), 3)
        print(
            'The image that best matches the given text is '
            f"{retrieval_ori_input['img_path']} and probability is {max_prob}")

        inputs = self.preprocess([retrieval_ori_input],
                                 pipeline=self.pipeline['grounding_pipeline'],
                                 batch_size=1,
                                 **preprocess_kwargs)

        self.model.task = 'ref-seg'
        self.model.sem_seg_head.task = 'ref-seg'
        self.model.sem_seg_head.predictor.task = 'ref-seg'

        ori_inputs, grounding_data = next(inputs)

        if isinstance(ori_inputs, dict):
            ori_inputs = ori_inputs['img_path']

        preds = self.forward(grounding_data, **forward_kwargs)

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
        if results['visualization'] is not None:
            results['visualization'] = results['visualization']
        return results
