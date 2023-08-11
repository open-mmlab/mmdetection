# Modified from MMPretrain
import gradio as gr
import torch
from mmengine.logging import MMLogger

from mmdet.apis import DetInferencer
from projects.XDecoder.xdecoder.inference import (
    ImageCaptionInferencer, RefImageCaptionInferencer,
    TextToImageRegionRetrievalInferencer)

logger = MMLogger('mmdetection', logger_name='mmdet')
if torch.cuda.is_available():
    gpus = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    logger.info(f'Available GPUs: {len(gpus)}')
else:
    gpus = None
    logger.info('No available GPU.')


def get_free_device():
    if gpus is None:
        return torch.device('cpu')
    if hasattr(torch.cuda, 'mem_get_info'):
        free = [torch.cuda.mem_get_info(gpu)[0] for gpu in gpus]
        select = max(zip(free, range(len(free))))[1]
    else:
        import random
        select = random.randint(0, len(gpus) - 1)
    return gpus[select]


class ObjectDetectionTab:
    model_list = [
        'retinanet_r50-caffe_fpn_1x_coco',
        'faster-rcnn_r50-caffe_fpn_1x_coco',
        'dino-5scale_swin-l_8xb2-12e_coco.py',
    ]

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input],
                    outputs=output,
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input], samples=[['demo/demo.jpg']])
            example_images.click(
                fn=lambda x: gr.Image.update(value=x[0]),
                inputs=example_images,
                outputs=image_input)

    def inference(self, model, image):
        det_inferencer = DetInferencer(
            model, scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(image, return_vis=True, no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class InstanceSegTab(ObjectDetectionTab):
    model_list = ['mask-rcnn_r50-caffe_fpn_1x_coco', 'solov2_r50_fpn_1x_coco']


class PanopticSegTab(ObjectDetectionTab):
    model_list = [
        'panoptic_fpn_r50_fpn_1x_coco',
        'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic'
    ]


class OpenVocabObjectDetectionTab:
    model_list = ['glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365']

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                text_input = gr.Textbox(
                    label='text prompt',
                    elem_classes='input_text',
                    interactive=True,
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input, text_input],
                    outputs=output,
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input, text_input],
                samples=[['demo/demo.jpg', 'bench . car .']])
            example_images.click(
                fn=self.update,
                inputs=example_images,
                outputs=[image_input, text_input])

    def update(self, example):
        return gr.Image.update(value=example[0]), gr.Textbox.update(
            value=example[1])

    def inference(self, model, image, text):
        det_inferencer = DetInferencer(
            model, scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(
            image,
            texts=text,
            custom_entities=True,
            pred_score_thr=0.5,
            return_vis=True,
            no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class GroundingDetectionTab(OpenVocabObjectDetectionTab):
    model_list = ['glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365']

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                text_input = gr.Textbox(
                    label='text prompt',
                    elem_classes='input_text',
                    interactive=True,
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input, text_input],
                    outputs=output,
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input, text_input],
                samples=[['demo/demo.jpg', 'There are a lot of cars here.']])
            example_images.click(
                fn=self.update,
                inputs=example_images,
                outputs=[image_input, text_input])

    def inference(self, model, image, text):
        det_inferencer = DetInferencer(
            model, scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(
            image,
            texts=text,
            custom_entities=False,
            pred_score_thr=0.5,
            return_vis=True,
            no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class OpenVocabInstanceSegTab(OpenVocabObjectDetectionTab):
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-instance_coco.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }

    def inference(self, model, image, text):
        det_inferencer = DetInferencer(
            **self.model_info[model], scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(
            image, texts=text, return_vis=True, no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class OpenVocabPanopticSegTab(OpenVocabObjectDetectionTab):
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-panoptic_coco.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='od_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Image',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                text_input = gr.Textbox(
                    label='thing text prompt',
                    elem_classes='input_text_thing',
                    interactive=True,
                )
                stuff_text_input = gr.Textbox(
                    label='stuff text prompt',
                    elem_classes='input_text_stuff',
                    interactive=True,
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[
                        select_model, image_input, text_input, stuff_text_input
                    ],
                    outputs=output,
                )
        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input, text_input, stuff_text_input],
                samples=[['demo/demo.jpg', 'bench.car', 'tree']])
            example_images.click(
                fn=self.update,
                inputs=example_images,
                outputs=[image_input, text_input, stuff_text_input])

    def update(self, example):
        return gr.Image.update(value=example[0]), \
            gr.Textbox.update(label='thing text prompt', value=example[1]), \
            gr.Textbox.update(label='stuff text prompt', value=example[2])

    def inference(self, model, image, text, stuff_text):
        det_inferencer = DetInferencer(
            **self.model_info[model], scope='mmdet', device=get_free_device())
        results_dict = det_inferencer(
            image,
            texts=text,
            stuff_texts=stuff_text,
            return_vis=True,
            no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class OpenVocabSemSegTab(OpenVocabInstanceSegTab):
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-semseg_coco.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }


class ReferSegTab(OpenVocabInstanceSegTab):
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_open-vocab-ref-seg_refcocog.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }


class ImageCaptionTab:
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_caption_coco2014.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_caption_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    interactive=True,
                    tool='editor',
                )
                caption_output = gr.Textbox(
                    label='Result',
                    lines=2,
                    elem_classes='caption_result',
                    interactive=False,
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input],
                    outputs=caption_output,
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input], samples=[['demo/demo.jpg']])
            example_images.click(
                fn=lambda x: gr.Image.update(value=x[0]),
                inputs=example_images,
                outputs=image_input)

    def inference(self, model, image):
        ic_inferencer = ImageCaptionInferencer(
            **self.model_info[model], scope='mmdet', device=get_free_device())
        results_dict = ic_inferencer(
            image, return_vis=False, no_save_vis=True, return_datasamples=True)
        return results_dict['predictions'][0].pred_caption


class ReferImageCaptionTab(OpenVocabInstanceSegTab):
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_ref-caption.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='image_caption_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                image_input = gr.Image(
                    label='Input',
                    source='upload',
                    elem_classes='input_image',
                    type='filepath',
                    interactive=True,
                    tool='editor',
                )
                text_input = gr.Textbox(
                    label='text prompt',
                    elem_classes='input_text',
                    interactive=True,
                )
                output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )
                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, image_input, text_input],
                    outputs=output,
                )

        with gr.Row():
            example_images = gr.Dataset(
                components=[image_input, text_input],
                samples=[['demo/demo.jpg', 'tree']])
            example_images.click(
                fn=self.update,
                inputs=example_images,
                outputs=[image_input, text_input])

    def update(self, example):
        return gr.Image.update(value=example[0]), gr.Textbox.update(
            value=example[1])

    def inference(self, model, image, text):
        ric_inferencer = RefImageCaptionInferencer(
            **self.model_info[model], scope='mmdet', device=get_free_device())
        results_dict = ric_inferencer(
            image, texts=text, return_vis=True, no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


class TextToImageRetrievalTab:
    model_list = ['xdecoder-tiny']

    model_info = {
        'xdecoder-tiny': {
            'model':
            'projects/XDecoder/configs/xdecoder-tiny_zeroshot_text-image-retrieval.py',  # noqa
            'weights':
            'https://download.openmmlab.com/mmdetection/v3.0/xdecoder/xdecoder_focalt_last_novg.pt'  # noqa
        }
    }

    def __init__(self) -> None:
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                select_model = gr.Dropdown(
                    label='Choose a model',
                    elem_id='t2i_retri_models',
                    elem_classes='select_model',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
            with gr.Column():
                prototype = gr.File(
                    file_count='multiple', file_types=['image'])
                text_input = gr.Textbox(
                    label='Query',
                    elem_classes='input_text',
                    interactive=True,
                )
                retri_output = gr.Image(
                    label='Result',
                    source='upload',
                    interactive=False,
                    elem_classes='result',
                )

                run_button = gr.Button(
                    'Run',
                    elem_classes='run_button',
                )
                run_button.click(
                    self.inference,
                    inputs=[select_model, prototype, text_input],
                    outputs=retri_output,
                )

    def inference(self, model, prototype, text):
        inputs = [file.name for file in prototype]
        retri_inferencer = TextToImageRegionRetrievalInferencer(
            **self.model_info[model], scope='mmdet', device=get_free_device())
        results_dict = retri_inferencer(
            inputs, texts=text, return_vis=True, no_save_vis=True)
        vis = results_dict['visualization'][0]
        return vis


if __name__ == '__main__':
    title = 'MMDetection Inference Demo'

    DESCRIPTION = '''# <div align="center">MMDetection Inference Demo  </div>
    <div align="center">
    <img src="https://user-images.githubusercontent.com/45811724/190993591-
    bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" width="50%"/>
    </div>

    #### This is an official demo for MMDet. \n

    - The first time running requires downloading the weights,
    please wait a moment. \n
    - OV is mean Open Vocabulary \n
    - Refer Seg is mean Referring Expression Segmentation \n
    - In Text-Image Region Retrieval, you need to provide n images and
    a query text, and the model will predict the most matching image and
    its corresponding grounding mask.
    '''

    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('Detection'):
                ObjectDetectionTab()
            with gr.TabItem('Instance'):
                InstanceSegTab()
            with gr.TabItem('Panoptic'):
                PanopticSegTab()
            with gr.TabItem('Grounding Detection'):
                GroundingDetectionTab()
            with gr.TabItem('OV Detection'):
                OpenVocabObjectDetectionTab()
            with gr.TabItem('OV Instance'):
                OpenVocabInstanceSegTab()
            with gr.TabItem('OV Panoptic'):
                OpenVocabPanopticSegTab()
            with gr.TabItem('OV SemSeg'):
                OpenVocabSemSegTab()
            with gr.TabItem('Refer Seg'):
                ReferSegTab()
            with gr.TabItem('Image Caption'):
                ImageCaptionTab()
            with gr.TabItem('Refer Caption'):
                ReferImageCaptionTab()
            with gr.TabItem('Text-Image Region Retrieval'):
                TextToImageRetrievalTab()
    demo.queue().launch(share=True)
