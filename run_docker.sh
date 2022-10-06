docker run \
    --gpus all \
    --shm-size=8g \
    --rm \
    -it \
    -v /home/chrissikek/repos/data:/mmdetection/data \
    -v /home/chrissikek/repos/mmdetection/:/mmdetection \
    -w /mmdetection \
    mmdetection