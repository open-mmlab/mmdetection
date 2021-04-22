#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # Downloading to current path 
	cd $(dirname $0)
    echo "Downloading VOC0712 to" $(pwd) "..." 
    
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "Downloading VOC0712 to" $1 "..."
    cd $1
fi

## Download the data
echo "Downloading VOCtrainval_06-Nov-2007.tar ..."
curl -LO http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

echo "Downloading VOCtest_06-Nov-2007.tar ..."
curl -LO http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

echo "Downloading VOCtrainval_11-May-2012.tar ..."
curl -LO http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

echo "Done downloading."

# Extract data
echo "Extracting VOCtrainval_11-May-2012.tar ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "Extracting VOCtrainval_06-Nov-2007.tar ..."
tar -xvf VOCtrainval_06-Nov-2007.tar
echo "Extracting VOCtest_06-Nov-2007.tar ..."
tar -xvf VOCtest_06-Nov-2007.tar

# Remove data
echo "Removing all tar ..."
rm VOCtrainval_11-May-2012.tar
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"