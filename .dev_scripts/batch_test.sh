export PYTHONPATH=${PWD}

partition=$1
model_dir=$2
json_out=$3
job_name=batch_test
gpus=8
gpu_per_node=8

touch $json_out
lastLine=$(tail -n 1 $json_out)
while [ "$lastLine" != "finished" ]
do
    srun -p ${partition} --gres=gpu:${gpu_per_node} -n${gpus} --ntasks-per-node=${gpu_per_node} \
        --job-name=${job_name} --kill-on-bad-exit=1 \
        python .dev_scripts/batch_test.py $model_dir $json_out --launcher='slurm'
    lastLine=$(tail -n 1 $json_out)
    echo $lastLine
done
