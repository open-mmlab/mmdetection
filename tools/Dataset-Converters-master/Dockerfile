FROM ubuntu:bionic as result

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends \
        build-essential python python-pip python-setuptools python-dev libsm6 libxrender1 libxext6 libglib2.0-0 \
 && pip install wheel

WORKDIR /opt
COPY dataset_converters dataset_converters
COPY LICENSE README.md *.py requirements.txt ./
RUN pip install -r requirements.txt


# Possible extension:
#ENTRYPOINT ["/usr/bin/python", "convert.py"]

# Usage examples:
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -qqy --no-install-recommends wget ca-certificates unzip \
# && wget --progress=dot -e dotbytes=100M http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip \
# && unzip -q ADE20K_2016_07_26.zip \
# && rm ADE20K_2016_07_26.zip

# Note, In order to run conversion it is recommended to have 8GB of RAM.

#RUN python convert.py -i ADE20K_2016_07_26 -o coco -I ADE20K -O COCO --symlink
#RUN python convert.py -i coco -o tdg -I COCO -O TDG --copy
