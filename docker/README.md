# How to run mmdetection via Docker
There are three steps to use this repository via Docker.

1. Build docker image
2. Run docker container 
3. Install dependencies

```bash
# build docker image
TAG_NAME=mmdetection
docker build -t $TAG_NAME .

# set mounted direcroty paths and enter container
DATADRIVE=/datadrive
MMDETECTION=/home/username/mmdetection
docker run --gpus all --shm-size=8g -it -v ${DATADRIVE}:/root/datadrive -v ${MMDETECTION}:/root/workspace ${TAG_NAME}

# inside container, install depended libraries
pip install --no-cache-dir -r requirements/build.txt
pip install --no-cache-dir -e .
```
