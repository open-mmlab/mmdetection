docker run \
    --gpus all \
    --shm-size=8g \
    --rm \
    -it \
    -v /data/brummer/mmdet_all_data:/mmdetection/data \
    -v /data/mmdetection/:/mmdetection \
    -v /data:/data \
    -w /mmdetection \
    mmdetection