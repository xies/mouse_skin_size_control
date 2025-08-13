**mouse_skin_size_control**

Scripts used reconstruct 3D cell shape of mouse epidermal stem cells from intravital imaging datasets.

This repository includes code used the following publications:

- ./cb_2020_analysis: Shicong Xie, Jan M Skotheim, A G1 sizer coordinates growth and division in the mouse epidermis. Current Biology 30 (5), 916-924. e2
DOI: https://doi.org/10.1016/j.cub.2019.12.062
- ./2024_analysis:  The G1/S transition in mammalian stem cells in vivo is autonomously regulated by cell size
Shicong Xie, Shuyuan Zhang, Gustavo de Medeiros, Prisca Liberali, Jan M. Skotheim. bioRxiv 2024.04.09.588781; doi: https://doi.org/10.1101/2024.04.09.588781 


----
**./2024_analysis**

Moved to: https://github.com/skotheimlab/xie_etal_2024_autonomous_cell_size_control

Tested on Python 3.9 on MacOS. More detailed instructions are in ./2024_analysis/instructions.txt. Requirements are listed in ./2024_analysis/requirements.txt


CONTENTS:

1) single cell tracking: Scripts for semi-automated movie assembly and collation of semi-automated single cell tracking in 3D

Expected input data: raw images, segmentation masks of nuclear shapes in 3D, and single cell tracking from MaMuT tracking tables

assemble_movie: semi-automated registration and assembly of movies from longitudinal snapshots
semiauto_tracking_segmentation: collate segmentation and sparse single cell tracking

2) cell and microenvironment quantifications: Scripts for quantifying and collating cell and cell-neighborhood geometries using densely annotated 3D cell and nuclear segmentations

Expected input data: raw images, segmentation masks of cell and nuclear shapes in 3D

annotate_tissue_dense: annotate cell and microenvironment features
tissue_dynamics_model: statistical models predicting cell cycle dynamics from cell or microenvironment features

