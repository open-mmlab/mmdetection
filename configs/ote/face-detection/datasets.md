## Prepare datasets

### 1. Download datasets

Download and unpack it to the `${DATA_DIR}` folder.
* WIDER Face Training Images - [WIDER_train.zip](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing)
* WIDER Face Validation Images - [WIDER_val.zip](https://drive.google.com/file/d/0B6eKvaijfFUDd3dIRmpvSk8tLUk/view?usp=sharing)
* Face annotations - [wider_face_split.zip](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

```bash
export DATA_DIR=${WORK_DIR}/data
```

So that `ls -la $DATA_DIR` gives you:
```
drwxrwxr-x 5 user user 4096 nov  3 12:42 .
drwxr-xr-x 8 user user 4096 nov  3 12:42 ..
drwxrwxr-x 2 user user 4096 nov 31  2017 wider_face_split
drwxrwxr-x 3 user user 4096 nov 18  2015 WIDER_train
drwxrwxr-x 3 user user 4096 nov 18  2015 WIDER_val
```

### 2. Convert datasets

Convert downloaded and extracted annotation to MSCOCO format with `face` as the only one class.

* Training annotation

   ```bash
   export TRAIN_ANN_FILE=${DATA_DIR}/instances_train.json
   export TRAIN_IMG_ROOT=${DATA_DIR}
   python ./configs/ote/face-detection/tools/wider_to_coco.py \
      ${DATA_DIR}/wider_face_split/wider_face_train_bbx_gt.txt \
      ${DATA_DIR}/WIDER_train/images/ \
      ${TRAIN_ANN_FILE}
   ```

* Validation annotation

   ```bash
   export VAL_ANN_FILE=${DATA_DIR}/instances_val.json
   export VAL_IMG_ROOT=${DATA_DIR}
   python ./configs/ote/face-detection/tools/wider_to_coco.py \
      ${DATA_DIR}/wider_face_split/wider_face_val_bbx_gt.txt \
      ${DATA_DIR}/WIDER_val/images/ \
      ${VAL_ANN_FILE}
   ```
