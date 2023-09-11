#!/bin/bash
set -euo pipefail

#present directory has to be home/user/mmdetection
dir_root=$(pwd)
data_dir=$(pwd)/data
datasets=$data_dir/rf100
benchmark=faster-RCNN-benchmark
config=$dir_root/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_rf100.py

cd $dir_root/projects/ROBOFLOW-100-BENCHMARK

if [ ! -d $datasets ] ; then
    $(pwd)/scripts/download_datasets.sh -l $datasets -f yolov5
fi

if [ ! -f "$(pwd)/runs/$benchmark/final_eval.txt" ] ; then
    touch "$(pwd)/runs/$benchmark/final_eval.txt"
fi

if [ ! -d "$(pwd)/runs/$benchmark/results" ] ; then
    mkdir -p "$(pwd)/runs/$benchmark/results"
fi


# fo rhttps://stackoverflow.com/questions/4011705/python-the-imagingft-c-module-is-not-installed
#apt-get install -y libfreetype6-dev
# setting up yolov8 - specific version, 20.01.2023
#pip install git+https://github.com/ultralytics/ultralytics.git@fix_shape_mismatch
# for AttributeError: partially initialized module ‘cv2’ has no attribute ‘gapi_wip_gst_GStreamerPipeline’ (most likely due to a circular import)
#pip install --force --no-cache-dir opencv-contrib-python==4.5.5.62 
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

num=0
for datas in $(ls $datasets)
do
    num=$((num+1))
    echo $num
    dataset=$datasets/$datas
    echo "Training on $dataset"
    data_json=$dataset/train/$(ls $dataset/train/ | grep json)
    echo "json address = $data_json"
    num_class=$(python $(pwd)/scripts/json_load.py --json $data_json --dataset $dataset --config $config)
    echo "class number = $num_class"

    # vim $config
    # echo $config

    if [ ! -d "$(pwd)/runs/$benchmark/results/$datas" ] ; then
        mkdir -p "$(pwd)/runs/$benchmark/results/$datas"
    fi

    if [ ! -d "$dataset/results" ] ;
    then
        bash $dir_root/tools/dist_train.sh $config 8 
        # --work-dir $(pwd)/runs/$benchmark/results/$datas
        # save best mAP in $date_final_eval.csv

        if [ $num -eq 1 ];then
        #     output=($(python $dir_root/projects/ROBOFLOW-100-BENCHMARK/scripts/log_extract.py $(pwd)/runs/$benchmark/results/$datas))
        #     log_dir=${output[0]}
        #     result_csv=${output[-1]}
        #     echo result_csv is saved as $result_csv
        #     #move ckpt to log dir
        #     cd $(pwd)/runs/$benchmark/results/$datas
        #     new=$(ls -td -- */ | head -n 1)
		#     echo $(ls)
		#     mv ./*.pth $new
		#     cd -
        #     echo "all checkpoints have been moved into directories!"
            read
        # else
        #     output=($(python $dir_root/projects/ROBOFLOW-100-BENCHMARK/scripts/log_extract.py $(pwd)/runs/$benchmark/results/$datas --result_csv $result_csv))
        #     log_dir=${output[0]}
        #     echo result has been saved in $result_csv

        #     cd $(pwd)/runs/$benchmark/results/$datas
        #     new=$(ls -td -- */ | head -n 1)
		#     echo $(ls)
		#     mv ./*.pth $new
		#     cd -
        fi
    fi
done
echo "Done training all the datasets with Faster-RCNN!"

# cd $dir_root/projects/ROBOFLOW-100-BENCHMARK/runs/$benchmark/results
# bash $dir_root/projects/ROBOFLOW-100-BENCHMARK/runs/$benchmark/results/ckpt
# echo "all checkpoints have been moved into directories!"





