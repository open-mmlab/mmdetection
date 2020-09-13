model_dir=$1
json_out=$2
touch $json_out
lastLine=$(tail -n 1 $json_out)
while [ "$lastLine" != "finished" ]
do
    ./tools/dist_batch_test.sh models res.json 8
    lastLine=$(tail -n 1 $json_out)
    echo $lastLine
done

# python .dev_scripts/batch_test.py models $json_out
