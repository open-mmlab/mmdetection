#!/bin/bash
VOLUME=$1

docker run --gpus all -it --name=mmdetection \
    --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -p 8082:22 -p 8083:6006 \
  -v $VOLUME:$VOLUME:rw \
  mmdetection_image

