#!/bin/bash

#SBATCH --job-name=cellpose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=batch
#SBATCH --account=skotheim
#SBATCH --time=02:00:00
#SBATCH --mem=500gb

module load anaconda
source activate cellpose

python /home/xies/Code/mouse_skin_size_control/live_analysis/cellpose_nuclear_seg/1__run_nuclei_model.py
