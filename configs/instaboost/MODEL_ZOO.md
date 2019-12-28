# Model Zoo

## Common settings
 - All models were trained on `coco_2017_train`, and tested on the `coco_2017_val`, for your conveinience of evaluation and comparison. In our paper, the numbers are obtained from test-dev.
 - To balance accuracy and training time when using InstaBoost, models released in this page are all trained for 48 Epochs. Other training and testing configs are strictly following the original framework. 
 - More results for other detection frameworks are avaliable [`here`](https://github.com/GothicAi/Instaboost).

## Baselines

|     Network     |       Backbone       | Lr schd |      box AP       |      mask AP       |      Download       |
| :-------------: |      :--------:      | :-----: |      :----:       |      :-----:       | :-----------------: |
|    Mask R-CNN   |       R-50-FPN       |   4x    |  39.90(orig:)  |  36.20(orig:)   |[Baidu]() / [Google]()|
