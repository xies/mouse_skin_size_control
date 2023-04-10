#!/bin/bash

#SBATCH --job-name=cp_nuc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=batch
#SBATCH --account=skotheim
#SBATCH --time=08:00:00
#SBATCH --mem=100gb

module load anaconda
source activate cellpose
which conda
conda env list

python /home/xies/Code/mouse_skin_size_control/live_analysis/cellpose_nuclear_seg/run_nuclei_model_scg.py
