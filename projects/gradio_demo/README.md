# MMDetection Gradio Demo

Here is a gradio demo for MMDetection supported inference tasks.

Currently supported tasks:

- Object Detection
- Instance Segmentation
- Panoptic Segmentation
- Grounding Object Detection
- Open Vocabulary Object Detection
- Open Vocabulary Instance Segmentation
- Open Vocabulary Semantic Segmentation
- Open Vocabulary Panoptic Segmentation
- Referring Expression Segmentation
- Image Caption
- Referring Expression Image Caption
- Text-To-Image Retrieval

## Preview

<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/421adfab-98cb-4b65-ab99-15d154cba95f" width="90%"/>

## Requirements

To run the demo, you need to install MMDetection at first. And please install with the extra multi-modality
dependencies to enable multi-modality tasks.

```shell
# At the MMDetection root folder
pip install -e ".[multimodal]"
```

And then install the latest gradio package.

```shell
pip install "gradio>=3.31.0"
```

## Start

Then, you can start the gradio server on the local machine by:

```shell
cd mmdetection
python projects/gradio_demo/launch.py
```

The demo will start a local server `http://127.0.0.1:7860` and you can browse it by your browser.
