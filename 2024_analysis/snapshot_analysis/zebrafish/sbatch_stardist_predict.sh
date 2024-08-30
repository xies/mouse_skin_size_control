#!/bin/bash

#SBATCH --job-name=stardist_pred
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=batch
#SBATCH --account=skotheim
#SBATCH --time=1:00:00
#SBATCH --mem=500gb

source activate stardist

python /home/xies/Code/mouse_skin_size_control/2024_analysis/snapshot_analysis/zebrafish/3d_predict.py
