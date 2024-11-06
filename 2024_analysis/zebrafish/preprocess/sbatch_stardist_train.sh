#!/bin/bash

#SBATCH --job-name=stardist_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=nih_s10
#SBATCH --account=skotheim
#SBATCH --time=3-00:00:00
#SBATCH --mem=100gb

nvidia-smi
source activate stardist

python /home/xies/Code/mouse_skin_size_control/2024_analysis/snapshot_analysis/zebrafish/3d_train.py
