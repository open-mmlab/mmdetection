import streamlit as st
from mmdet.apis import init_detector, inference_detector
import mmcv

st.title("Industrial Defect Detector")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    image = mmcv.imread(uploaded_file)
    model = init_detector('configs/defect/faster-rcnn_r50_fpn_1x_neu.py', 'checkpoints/latest.pth')
    result = inference_detector(model, image)
    model.show_result(image, result, out_file='output.jpg')
    st.image('output.jpg')
