#!/bin/bash
VOLUME=$1

docker run --gpus all -it --name=mmdetection \
    --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -p 8084:22 -p 8085:6006 \
  -v $VOLUME:$VOLUME:rw \
  mmdetection_image

