## Pascal voc data format to coco data format
Note that this script does not support the integration of instance segmentation annotations.
```
python tools/misc/coco_split.py --xml_dir ${Directory path to xml files} \
                                --json_file ${Output COCO format json file} \
```
For example:
```
python tools/misc/coco_split.py --xml_dir ./data/annotations \
                                --json_file ./data \
```
## Divide dataset into training set, validation set and test set

Usually, custom dataset is a large folder with full of images. We need to divide the dataset into training set, validation set and test set by ourselves. If the amount of data is small, we can not divide the validation set. Here’s how the split script works:

```
python tools/misc/coco_train_split.py --json ${COCO label json path} \
                                      --out-dir ${divide label json saved path} \
                                      --ratios ${ratio of division} \
                                      [--shuffle] \
                                      [--seed ${random seed for division}]
```



These include:

- `--ratios`: ratio of division. If only 2 are set, the split is `trainval + test`, and if 3 are set, the split is `train + val + test`. Two formats are supported - integer and decimal:
  - Integer: divide the dataset in proportion after normalization. Example: `--ratio 2 1 1` (the code will convert to `0.5 0.25 0.25`) or `--ratio 3 1`（the code will convert to `0.75 0.25`）
  - Decimal: divide the dataset in proportion. **If the sum does not add up to 1, the script performs an automatic normalization correction.** Example: `--ratio 0.8 0.1 0.1` or `--ratio 0.8 0.2`
- `--shuffle`: whether to shuffle the dataset before splitting.
- `--seed`: the random seed of dataset division. If not set, this will be generated automatically.

For example:

```
python tools/misc/coco_split.py --json ./data/cat/annotations/annotations_all.json \
                                --out-dir ./data/cat/annotations \
                                --ratios 0.8 0.2 \
                                --shuffle \
                                --seed 10
```