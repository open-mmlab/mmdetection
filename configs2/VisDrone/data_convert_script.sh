sahi coco slice --image_dir data/VisDrone2019/VisDrone2019-DET-val/images \
  --dataset_json_path data/VisDrone2019/val.json \
  --slice_size=640 --overlap_ratio=0.0625 \
  --output_dir=data/VisDrone2019/VisDrone2019_slice640_overlap40

sahi coco slice --image_dir data/VisDrone2019/VisDrone2019-DET-train/images \
  --dataset_json_path data/VisDrone2019/train.json \
  --slice_size=640 --overlap_ratio=0.0625 \
  --output_dir=data/VisDrone2019/VisDrone2019_slice640_overlap40

sahi coco slice --image_dir data/VisDrone2019/VisDrone2019-DET-test-dev/images \
  --dataset_json_path data/VisDrone2019/test-dev.json \
  --slice_size=640 --overlap_ratio=0.0625 \
  --output_dir=data/VisDrone2019/VisDrone2019_slice640_overlap40

# 测试转换后的数据集是否对上了
python  tools/misc/browse_dataset.py   configs2/VisDrone/base/faster_rcnn_r50_fpn_1x_VisDrone640.py

