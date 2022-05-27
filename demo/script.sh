 python demo/webcam_demo.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py .\checkpoints\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

 python demo/video_demo.py demo/demo.mp4 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py .\checkpoints\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --show

 python demo/my_image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --async-test --score-thr 0.7 --save-path cache/cache.jpg