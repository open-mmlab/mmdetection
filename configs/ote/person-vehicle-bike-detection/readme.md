# Person-Vehicle-Bike Detector

The crossroad-detection network model provides detection of three class objects: vehicle, pedestrian, non-vehicle (like bikes). This detector was trained on the data from crossroad cameras.

| Model Name | Complexity (GFLOPs) | Size (Mp) | AP @ [IoU=0.50:0.95] (%) | Links | GPU_NUM |
| --- | --- | --- | --- | --- | --- |
| person-vehicle-bike-detection-2000 | 0.82 | 1.84 | 16.5 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2000-1.pth), [config](./person-vehicle-bike-detection-2000/template.yaml) | 4 |
| person-vehicle-bike-detection-2001 | 1.86 | 1.84 | 22.6 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2001-1.pth), [config](./person-vehicle-bike-detection-2001/template.yaml) | 4 |
| person-vehicle-bike-detection-2002 | 3.30 | 1.84 | 24.8 | [snapshot](https://download.01.org/opencv/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2002-1.pth), [config](./person-vehicle-bike-detection-2002/template.yaml) | 4 |
| person-vehicle-bike-detection-2003 | 6.78 | 1.95 | 33.6 | [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2003.pth), [config](./person-vehicle-bike-detection-2003/template.yaml) | 2 |
| person-vehicle-bike-detection-2004 | 1.88 | 1.95 | 27.4 | [snapshot](https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/vehicle-person-bike-detection-2004.pth), [config](./person-vehicle-bike-detection-2004/template.yaml) | 2 |
