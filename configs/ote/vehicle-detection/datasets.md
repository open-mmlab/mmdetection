## Prepare datasets

### 1. Download datasets

Collect or download images with vehicles presented on them. One can download MS-COCO dataset and remain images with cars only.
```bash
export DATA_DIR=${WORK_DIR}/data
wget http://images.cocodataset.org/zips/val2017.zip -P ${DATA_DIR}/
wget http://images.cocodataset.org/zips/train2017.zip -P ${DATA_DIR}/
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ${DATA_DIR}/
unzip ${DATA_DIR}/val2017.zip -d ${DATA_DIR}/
unzip ${DATA_DIR}/train2017.zip -d ${DATA_DIR}/
unzip ${DATA_DIR}/annotations_trainval2017.zip -d ${DATA_DIR}/
python tools/coco_filter.py ${DATA_DIR}/annotations/instances_train2017.json ${DATA_DIR}/annotations/instances_train2017car.json --filter car --remap
python tools/coco_filter.py ${DATA_DIR}/annotations/instances_val2017.json ${DATA_DIR}/annotations/instances_val2017car.json --filter car --remap
sed -i "s/car/vehicle/g" ${DATA_DIR}/annotations/instances_val2017car.json
sed -i "s/car/vehicle/g" ${DATA_DIR}/annotations/instances_train2017car.json
```
