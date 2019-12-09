#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A bcam
#SBATCH  
#SBATCH -J faster-rcnn-iou-ce
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --workdir=/truba_scratch/bcam/output/lrp_optimization
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bariscancam@gmail.com
#SBATCH --time=0-8:00:00
#SBATCH --output=/truba_scratch/bcam/output/lrp_optimization/faster-rcnn-iou-ce.out
#SBATCH --error=/truba_scratch/bcam/output/lrp_optimization/faster-rcnn-iou-ce.err

###################  Init. CUDA ##########################
export CUDA_HOME=/truba_scratch/bcam/software/cuda-10.0
export PATH=/truba_scratch/bcam/software/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/bcam/software/cuda-10.0/lib64
######################################################################

###################  Init. conda ##########################
_conda_setup="$('/truba_scratch/bcam/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/truba_scratch/bcam/software/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/truba_scratch/bcam/software/miniconda3/etc/profile.d/conda.sh"
    else
	export PATH="/truba_scratch/bcam/software/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
#export PATH="/truba_scratch/bcam/software/miniconda3/condabin:$PATH"
######################################################################

###################  Set open-mmlab env ##########################
source /truba_scratch/bcam/software/miniconda3/bin/activate open-mmlab
######################################################################

###################  Run ##########################
cd /truba/home/bcam/imgworkspace/mmdetection
python tools/train.py configs/lrp_optimization/initial_experiments/faster_rcnn_r50_fpn_1x_IoU_CE.py --gpus 2
######################################################################
