#!/bin/bash
#set -euo pipefail
input="$(pwd)/scripts/datasets_links_640.txt"

while getopts f:l: flag
do
    case "${flag}" in
        f) format=${OPTARG};;
        l) location=${OPTARG};;
    esac
done
# default values
format=${format:-coco}
location=${location:-$(pwd)/rf100}

echo "Starting downloading RF100..."

for link in $(cat $input)
do
    attributes=$(python3 $(pwd)/scripts/parse_dataset_link.py  -l $link)

    project=$(echo $attributes | cut -d' ' -f 3)
    version=$(echo $attributes | cut -d' ' -f 4)
    if [ ! -d  "$location/$project" ] ;
    then
        python3 $(pwd)/scripts/download_dataset.py -p $project -v $version -l $location -f $format
    fi
done

echo "Done!"
