from pytriton.client import ModelClient
import cv2
import numpy as np
def inference_service(file):
    input1_data = cv2.imread(file)
    input1_data = np.expand_dims(input1_data, axis=0)
    with ModelClient("localhost:8000", "MMDetectionModel") as client:
        result_dict = client.infer_batch(input1_data)
        print(result_dict)


